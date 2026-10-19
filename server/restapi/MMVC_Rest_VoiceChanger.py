import base64
import struct
import numpy as np
import traceback

from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from voice_changer.VoiceChangerManager import VoiceChangerManager
from pydantic import BaseModel
import threading
import resampy
from mods.log_control import VoiceChangaerLogger

logger = VoiceChangaerLogger.get_instance().getLogger()


class VoiceModel(BaseModel):
    timestamp: int
    buffer: str


class CalibrationRequest(BaseModel):
    sampleRate: int
    noise: str  # base64 int16 mono: the room with the speaker silent
    voice: str  # base64 int16 mono: the speaker reading aloud
    target: str = "keep"  # female | male | keep


class ClipRequest(BaseModel):
    sampleRate: int
    voice: str  # base64 int16 mono


def _decode_pcm(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    return np.frombuffer(raw[: len(raw) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0


class MMVC_Rest_VoiceChanger:
    def __init__(self, voiceChangerManager: VoiceChangerManager):
        self.voiceChangerManager = voiceChangerManager
        self.router = APIRouter()
        self.router.add_api_route("/test", self.test, methods=["POST"])
        self.router.add_api_route("/calibrate", self.calibrate, methods=["POST"])
        self.router.add_api_route("/convert_clip", self.convert_clip, methods=["POST"])

        self.tlock = threading.Lock()

    def test(self, voice: VoiceModel):
        try:
            timestamp = voice.timestamp
            buffer = voice.buffer
            wav = base64.b64decode(buffer)

            # if wav == 0:
            #     samplerate, data = read("dummy.wav")
            #     unpackedData = data
            # else:
            #     unpackedData = np.array(
            #         struct.unpack("<%sh" % (len(wav) // struct.calcsize("<h")), wav)
            #     )

            unpackedData = np.array(struct.unpack("<%sh" % (len(wav) // struct.calcsize("<h")), wav)).astype(np.int16)
            # print(f"[REST] unpackedDataType {unpackedData.dtype}")

            self.tlock.acquire()
            changedVoice = self.voiceChangerManager.changeVoice(unpackedData)
            self.tlock.release()

            changedVoiceBase64 = base64.b64encode(changedVoice[0]).decode("utf-8")
            data = {"timestamp": timestamp, "changedVoiceBase64": changedVoiceBase64}

            json_compatible_item_data = jsonable_encoder(data)
            return JSONResponse(content=json_compatible_item_data)

        except Exception as e:
            print("REQUEST PROCESSING!!!! EXCEPTION!!!", e)
            print(traceback.format_exc())
            self.tlock.release()
            return str(e)

    def _pitch_fn(self):
        """RVC's own pitch extractor when an RVC voice is loaded, so the suggested shift matches
        what conversion will see."""
        model = getattr(self.voiceChangerManager, "voiceChangerModel", None)
        pipeline = getattr(model, "pipeline", None)
        extractor = getattr(pipeline, "pitchExtractor", None)
        if extractor is None:
            return None

        def fn(x):
            _, pitchf = extractor.extract(x, np.zeros(x.shape[0] // 160 + 1, dtype=np.float32), 0, 16000, 160)
            return np.asarray(pitchf.cpu() if hasattr(pitchf, "cpu") else pitchf, dtype=np.float64)

        return fn

    def calibrate(self, req: CalibrationRequest):
        import importlib.util
        from voice_changer.common.robust.Calibration import calibrate, estimate_pitch

        try:
            noise = resampy.resample(_decode_pcm(req.noise), req.sampleRate, 16000)
            voice = resampy.resample(_decode_pcm(req.voice), req.sampleRate, 16000)
            pitch_fn = self._pitch_fn() or estimate_pitch
            try:
                result = calibrate(noise, voice, req.target, importlib.util.find_spec("df") is not None, pitch_fn)
            except Exception:
                logger.warning(f"[Calibration] model pitch extractor failed, using YIN: {traceback.format_exc()}")
                result = calibrate(noise, voice, req.target, importlib.util.find_spec("df") is not None, estimate_pitch)
            return JSONResponse(content=jsonable_encoder(result))
        except Exception as e:
            logger.error(traceback.format_exc())
            return JSONResponse(content={"error": str(e)}, status_code=400)

    def convert_clip(self, req: ClipRequest):
        """Converts a recording block by block, as live conversion would, so it can be heard
        before and after. Live conversion should be stopped while this runs."""
        try:
            vc = self.voiceChangerManager.voiceChanger
            if vc is None:
                return JSONResponse(content={"error": "no voice loaded"}, status_code=400)
            in_sr, out_sr = vc.settings.inputSampleRate, vc.settings.outputSampleRate
            voice = resampy.resample(_decode_pcm(req.voice), req.sampleRate, in_sr)
            pcm = (np.clip(voice, -1, 1) * 32767).astype(np.int16)
            block = 4096
            pcm = np.concatenate([pcm, np.zeros(block * 4, dtype=np.int16)])  # flush the latency
            out = []
            with self.tlock:
                for i in range(0, pcm.shape[0] - block + 1, block):
                    changed = self.voiceChangerManager.changeVoice(pcm[i : i + block])[0]
                    out.append(np.asarray(changed, dtype=np.int16))
            audio = np.concatenate(out) if out else np.zeros(0, dtype=np.int16)
            data = {"sampleRate": out_sr, "audio": base64.b64encode(audio.tobytes()).decode("utf-8")}
            return JSONResponse(content=jsonable_encoder(data))
        except Exception as e:
            logger.error(traceback.format_exc())
            return JSONResponse(content={"error": str(e)}, status_code=400)
