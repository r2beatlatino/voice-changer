import os
from concurrent.futures import ThreadPoolExecutor

from downloader.Downloader import download, is_usable
from mods.log_control import VoiceChangaerLogger
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams
from Exceptions import WeightDownladException

logger = VoiceChangaerLogger.get_instance().getLogger()


def downloadWeight(voiceChangerParams: VoiceChangerParams):
    content_vec_500_onnx = voiceChangerParams.content_vec_500_onnx
    hubert_base = voiceChangerParams.hubert_base
    hubert_base_jp = voiceChangerParams.hubert_base_jp
    hubert_soft = voiceChangerParams.hubert_soft
    nsf_hifigan = voiceChangerParams.nsf_hifigan
    crepe_onnx_full = voiceChangerParams.crepe_onnx_full
    crepe_onnx_tiny = voiceChangerParams.crepe_onnx_tiny
    rmvpe = voiceChangerParams.rmvpe
    rmvpe_onnx = voiceChangerParams.rmvpe_onnx
    whisper_tiny = voiceChangerParams.whisper_tiny

    weight_files = [
        content_vec_500_onnx,
        hubert_base,
        hubert_base_jp,
        hubert_soft,
        nsf_hifigan,
        crepe_onnx_full,
        crepe_onnx_tiny,
        rmvpe,
        whisper_tiny,
    ]

    # file exists check (currently only for rvc)
    downloadParams = []
    if not is_usable(hubert_base):
        downloadParams.append(
            {
                "url": "https://huggingface.co/ddPn08/rvc-webui-models/resolve/main/embeddings/hubert_base.pt",
                "saveTo": hubert_base,
                "position": 0,
            }
        )
    if not is_usable(hubert_base_jp):
        downloadParams.append(
            {
                "url": "https://huggingface.co/rinna/japanese-hubert-base/resolve/main/fairseq/model.pt",
                "saveTo": hubert_base_jp,
                "position": 1,
            }
        )
    if not is_usable(hubert_soft):
        downloadParams.append(
            {
                "url": "https://huggingface.co/wok000/weights/resolve/main/ddsp-svc30/embedder/hubert-soft-0d54a1f4.pt",
                "saveTo": hubert_soft,
                "position": 2,
            }
        )
    if not is_usable(nsf_hifigan):
        downloadParams.append(
            {
                "url": "https://huggingface.co/wok000/weights/resolve/main/ddsp-svc30/nsf_hifigan_20221211/model.bin",
                "saveTo": nsf_hifigan,
                "position": 3,
            }
        )
    nsf_hifigan_config = os.path.join(os.path.dirname(nsf_hifigan), "config.json")

    if not is_usable(nsf_hifigan_config):
        downloadParams.append(
            {
                "url": "https://huggingface.co/wok000/weights/raw/main/ddsp-svc30/nsf_hifigan_20221211/config.json",
                "saveTo": nsf_hifigan_config,
                "position": 4,
            }
        )
    nsf_hifigan_onnx = os.path.join(os.path.dirname(nsf_hifigan), "nsf_hifigan.onnx")
    if not is_usable(nsf_hifigan_onnx):
        downloadParams.append(
            {
                "url": "https://huggingface.co/wok000/weights/resolve/main/ddsp-svc30/nsf_hifigan_onnx_20221211/nsf_hifigan.onnx",
                "saveTo": nsf_hifigan_onnx,
                "position": 4,
            }
        )

    if not is_usable(crepe_onnx_full):
        downloadParams.append(
            {
                "url": "https://huggingface.co/wok000/weights/resolve/main/crepe/onnx/full.onnx",
                "saveTo": crepe_onnx_full,
                "position": 5,
            }
        )
    if not is_usable(crepe_onnx_tiny):
        downloadParams.append(
            {
                "url": "https://huggingface.co/wok000/weights/resolve/main/crepe/onnx/tiny.onnx",
                "saveTo": crepe_onnx_tiny,
                "position": 6,
            }
        )

    if not is_usable(content_vec_500_onnx):
        downloadParams.append(
            {
                "url": "https://huggingface.co/wok000/weights_gpl/resolve/main/content-vec/contentvec-f.onnx",
                "saveTo": content_vec_500_onnx,
                "position": 7,
            }
        )
    if not is_usable(rmvpe):
        downloadParams.append(
            {
                "url": "https://huggingface.co/wok000/weights/resolve/main/rmvpe/rmvpe_20231006.pt",
                "saveTo": rmvpe,
                "position": 8,
            }
        )
    if not is_usable(rmvpe_onnx):
        downloadParams.append(
            {
                "url": "https://huggingface.co/wok000/weights_gpl/resolve/main/rmvpe/rmvpe_20231006.onnx",
                "saveTo": rmvpe_onnx,
                "position": 9,
            }
        )

    if not is_usable(whisper_tiny):
        downloadParams.append(
            {
                "url": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
                "saveTo": whisper_tiny,
                "position": 10,
            }
        )

    with ThreadPoolExecutor() as pool:
        pool.map(download, downloadParams)

    if not all(is_usable(w) for w in [hubert_base, hubert_base_jp, hubert_soft, nsf_hifigan, nsf_hifigan_config]):
        raise WeightDownladException()

    # ファイルサイズをログに書き込む。（デバッグ用）
    for weight in weight_files:
        if is_usable(weight):
            file_size = os.path.getsize(weight)
            logger.debug(f"weight file [{weight}]: {file_size}")
        else:
            logger.warning(f"weight file is missing. {weight}")
            raise WeightDownladException()
