from dataclasses import dataclass, field


@dataclass
class RVCSettings:
    gpu: int = -9999
    dstId: int = 0

    f0Detector: str = "rmvpe_onnx"  # dio or harvest
    tran: int = 12
    silentThreshold: float = 0.00001
    extraConvertSize: int = 1024 * 4

    indexRatio: float = 0
    protect: float = 0.5
    rvcQuality: int = 0
    silenceFront: int = 1  # 0:off, 1:on
    modelSamplingRate: int = 48000

    # noise robustness (server side, works in server-audio mode too)
    denoise: int = 1  # 0:off, 1:GTCRN
    denoiseStrength: float = 1.0  # 0.0 (raw mic) .. 1.0 (fully denoised)
    vad: int = 1  # 0:off, 1:Silero VAD gate
    vadThreshold: float = 0.3
    vadHangoverMs: int = 500  # keep the gate open this long after speech ends
    f0Stabilize: int = 1  # 0:off, 1:remove pitch blips, gaps and octave jumps

    speakers: dict[str, int] = field(default_factory=lambda: {})
    # isHalf: int = 1  # 0:off, 1:on
    # enableDirectML: int = 0  # 0:off, 1:on
    # ↓mutableな物だけ列挙
    intData = [
        "gpu",
        "dstId",
        "tran",
        "extraConvertSize",
        "rvcQuality",
        "silenceFront",
        "denoise",
        "vad",
        "vadHangoverMs",
        "f0Stabilize",
    ]
    floatData = ["silentThreshold", "indexRatio", "protect", "denoiseStrength", "vadThreshold"]
    strData = ["f0Detector"]
