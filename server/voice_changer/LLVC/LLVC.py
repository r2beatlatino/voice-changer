import traceback
from typing import Any, cast
from scipy import signal
import os
from dataclasses import dataclass, asdict, field
import resampy
from data.ModelSlot import LLVCModelSlot
from mods.log_control import VoiceChangaerLogger
import numpy as np
from voice_changer.LLVC.LLVCInferencer import LLVCInferencer
from voice_changer.ModelSlotManager import ModelSlotManager
from voice_changer.VoiceChangerParamsManager import VoiceChangerParamsManager
from voice_changer.utils.Timer import Timer2
from voice_changer.utils.VoiceChangerModel import AudioInOut, AudioInOutFloat, VoiceChangerModel
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams
import math
import torchaudio
import torch

logger = VoiceChangaerLogger.get_instance().getLogger()


@dataclass
class LLVCSetting:
    # Crossfade(CF), Resample(RE) 組み合わせ
    # CF:True, RE:True -> ブラウザで使える
    # CF:True, RE:False -> N/A, 必要のない設定。（Resampleしないと音はぶつぶつしない。)
    # CF:False, RE:True -> N/A, 音にぷつぷつが乗るのでNG（client, server両モードでNGだった）
    # CF:False, RE:False -> 再生側が16Kに対応していればよい。

    crossfade: bool = True
    resample: bool = True

    # 変更可能な変数だけ列挙
    intData: list[str] = field(default_factory=lambda: [])
    floatData: list[str] = field(default_factory=lambda: [])
    strData: list[str] = field(default_factory=lambda: [])


class StreamResampler:
    """Chunk-by-chunk resampling without edge artefacts: each call resamples with some context on
    both sides and emits only the middle, so the output is delayed by that context (about 1-4 ms).
    It only cuts at input positions that map to whole output samples, or the chunks would be
    misaligned by a fraction of a sample."""

    MIN_CTX = 64

    def __init__(self, sr_in: int, sr_out: int):
        self.sr_in = sr_in
        self.sr_out = sr_out
        d = math.gcd(sr_in, sr_out)
        self.step_in = sr_in // d
        self.step_out = sr_out // d
        self.ctx = math.ceil(self.MIN_CTX / self.step_in) * self.step_in
        self.buf = np.zeros(self.ctx, dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.buf = np.concatenate([self.buf, x.astype(np.float32)])
        emit = (self.buf.shape[0] - 2 * self.ctx) // self.step_in * self.step_in
        if emit <= 0:
            return np.zeros(0, dtype=np.float32)
        y = self.buf if self.sr_in == self.sr_out else resampy.resample(self.buf, self.sr_in, self.sr_out, filter="kaiser_fast")
        start = self.ctx // self.step_in * self.step_out
        out = y[start : start + emit // self.step_in * self.step_out]
        self.buf = self.buf[emit:]
        return out.astype(np.float32)


class LLVC(VoiceChangerModel):
    def __init__(self, params: VoiceChangerParams, slotInfo: LLVCModelSlot):
        logger.info("[Voice Changer] [LLVC] Creating instance ")
        self.voiceChangerType = "LLVC"
        self.settings = LLVCSetting()

        self.processingSampleRate = 16000
        bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=self.processingSampleRate)
        self.bh = bh
        self.ah = ah

        self.params = params
        self.slotInfo = slotInfo
        self.modelSlotManager = ModelSlotManager.get_instance(self.params.model_dir)

        # # クロスフェード・リサンプリング設定
        # ## 16Kで出力するモード
        # self.settings.crossfade = False
        # self.settings.resample = False

        ## 48Kで出力するモード
        self.settings.crossfade = True
        self.settings.resample = True

        self.initialize()

    def initialize(self):
        print("[Voice Changer] [LLVC] Initializing... ")
        vcparams = VoiceChangerParamsManager.get_instance().params
        configPath = os.path.join(vcparams.model_dir, str(self.slotInfo.slotIndex), self.slotInfo.configFile)
        modelPath = os.path.join(vcparams.model_dir, str(self.slotInfo.slotIndex), self.slotInfo.modelFile)

        self.inputSampleRate = 48000
        self.outputSampleRate = 48000

        self.downsampler = torchaudio.transforms.Resample(self.inputSampleRate, self.processingSampleRate)
        self.upsampler = torchaudio.transforms.Resample(self.processingSampleRate, self.outputSampleRate)

        self.inferencer = LLVCInferencer().loadModel(modelPath, configPath)
        self.prev_audio1 = None
        self.result_buff = None
        self._reset_stream()

    def _reset_stream(self):
        self.in_rs = StreamResampler(self.inputSampleRate, self.processingSampleRate)
        # one processing unit of headroom so the FIFO below never runs dry
        unit = self.inferencer.model.dec_chunk_size * self.inferencer.model.L
        self.out_fifo = np.zeros(unit, dtype=np.float32)
        self.in_total = 0
        self.out_total = 0
        self.hp_sos = signal.butter(N=5, Wn=48, btype="high", fs=self.processingSampleRate, output="sos")
        self.hp_zi = np.zeros((self.hp_sos.shape[0], 2))

    def updateSetting(self, key: str, val: Any):
        if key in self.settings.intData:
            setattr(self.settings, key, int(val))
            ret = True
        elif key in self.settings.floatData:
            setattr(self.settings, key, float(val))
            ret = True
        elif key in self.settings.strData:
            setattr(self.settings, key, str(val))
            ret = True
        else:
            ret = False

        return ret

    def setSamplingRate(self, inputSampleRate, outputSampleRate):
        self.inputSampleRate = inputSampleRate
        self.outputSampleRate = outputSampleRate
        self.downsampler = torchaudio.transforms.Resample(self.inputSampleRate, self.processingSampleRate)
        self.upsampler = torchaudio.transforms.Resample(self.processingSampleRate, self.outputSampleRate)
        if hasattr(self, "inferencer"):
            self._reset_stream()

    def _preprocess(self, waveform: AudioInOutFloat, srcSampleRate: int) -> AudioInOutFloat:
        """データ前処理(torch independent)
        ・マルチディメンション処理
        ・リサンプリング( 入力sr -> 16K)
        ・バターフィルタ
        Args:
            waveform: AudioInOutFloat: 入力音声
            srcSampleRate: int: 入力音声のサンプルレート

        Returns:
            waveform: AudioInOutFloat: 前処理後の音声(1ch, 16K, np.ndarray)

        Raises:
            OSError: ファイル指定が失敗している場合

        """
        if waveform.ndim == 2:  # double channels
            waveform = waveform.mean(axis=-1)
        waveform16K = resampy.resample(waveform, srcSampleRate, self.processingSampleRate)
        # waveform16K = self.downsampler(torch.from_numpy(waveform)).numpy()
        waveform16K = signal.filtfilt(self.bh, self.ah, waveform16K)
        return waveform16K.copy()

    def inference(self, receivedData: AudioInOut, crossfade_frame: int, sola_search_frame: int):
        try:
            # print("CROSSFADE", crossfade_frame, sola_search_frame)
            crossfade_frame16k = math.ceil((crossfade_frame / self.outputSampleRate) * self.processingSampleRate)
            sola_search_frame16k = math.ceil((sola_search_frame / self.outputSampleRate) * self.processingSampleRate)

            with Timer2("mainPorcess timer", False) as t:
                # 起動パラメータ
                # vcParams = VoiceChangerParamsManager.get_instance().params

                # リサンプリングとバターフィルタ (torch independent)
                receivedData = receivedData.astype(np.float32) / 32768.0
                if receivedData.ndim == 2:
                    receivedData = receivedData.mean(axis=-1)
                # streaming resample + causal high-pass: per-chunk resampy and filtfilt left edge
                # artefacts at every block boundary
                waveformFloat = self.in_rs(receivedData)
                waveformFloat, self.hp_zi = signal.sosfilt(self.hp_sos, waveformFloat, zi=self.hp_zi)

                # 推論
                audio1 = self.inferencer.infer(waveformFloat.astype(np.float32))
                audio1 = audio1.detach().cpu().numpy()

                # The model only emits whole processing units, so its output length jitters from
                # block to block. SOLA expects exactly one block (plus overlap) per call, otherwise
                # it re-uses or skips audio, so hand it a fixed amount from a FIFO.
                self.out_fifo = np.concatenate([self.out_fifo, audio1])
                self.in_total += receivedData.shape[0]
                need = round(self.in_total * self.processingSampleRate / self.inputSampleRate) - self.out_total
                if self.out_fifo.shape[0] < need:
                    self.out_fifo = np.concatenate([self.out_fifo, np.zeros(need - self.out_fifo.shape[0], dtype=np.float32)])
                audio1, self.out_fifo = self.out_fifo[:need], self.out_fifo[need:]
                self.out_total += need
                # print(f"infered shape: in:{waveformFloat.shape} -> out:{ audio1.shape}")

                # クロスフェード洋データ追加とリサンプリング
                if self.settings.crossfade is False and self.settings.resample is False:
                    # 変換後そのまま返却(クロスフェードしない)
                    new_audio = audio1
                    new_audio = (new_audio * 32767.5).astype(np.int16)
                    return new_audio

                # (1) クロスフェード部分の追加
                crossfade_audio_length = audio1.shape[0] + crossfade_frame16k + sola_search_frame16k
                if self.prev_audio1 is not None:
                    new_audio = np.concatenate([self.prev_audio1, audio1])
                else:
                    new_audio = audio1
                self.prev_audio1 = new_audio[-crossfade_audio_length:]  # 次回のクロスフェード用に保存
                # (2) リサンプル
                if self.outputSampleRate != self.processingSampleRate:
                    new_audio = resampy.resample(new_audio, self.processingSampleRate, self.outputSampleRate)
                    # new_audio = self.upsampler(torch.from_numpy(new_audio)).numpy()
                    # new_audio = np.repeat(new_audio, 3)

                # バッファリング。⇒ 最上位(crossfade完了後)で行う必要があるのでとりあえずペンディング
                # if self.result_buff is None:
                #     self.result_buff = new_audio
                # else:
                #     self.result_buff = np.concatenate([self.result_buff, new_audio])

                # if self.result_buff.shape[0] > receivedData.shape[0]:
                #     new_audio = self.result_buff[: receivedData.shape[0]]
                #     self.result_buff = self.result_buff[receivedData.shape[0] :]
                # else:
                #     new_audio = np.zeros(receivedData.shape[0])

                new_audio = cast(AudioInOutFloat, new_audio)

                new_audio = (new_audio * 32767.5).astype(np.int16)
                return new_audio
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(e)

    def getPipelineInfo(self):
        return {"TODO": "LLVC get info"}

    def get_info(self):
        data = asdict(self.settings)

        return data

    def get_processing_sampling_rate(self):
        return self.processingSampleRate

    def get_model_current(self):
        return []
