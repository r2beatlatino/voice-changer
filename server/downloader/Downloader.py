import requests  # type: ignore
import os
import time
from tqdm import tqdm

from mods.log_control import VoiceChangaerLogger

logger = VoiceChangaerLogger.get_instance().getLogger()

# give up after this many attempts in a row that brought no new bytes; big files on a flaky
# connection are cut every few MB, so attempts that make progress don't count
MAX_STALLED_TRIES = 10
MAX_SECONDS = 45 * 60
CHUNK = 8192  # small, so little is lost when a connection drops mid-chunk
MODEL_EXTENSIONS = (".pt", ".pth", ".onnx", ".bin", ".safetensors")


def is_usable(path: str) -> bool:
    """A model file of a few bytes is an error page or an old failed download, not the model."""
    if not os.path.exists(path):
        return False
    return not path.lower().endswith(MODEL_EXTENSIONS) or os.path.getsize(path) >= 1024


def _fetch(url: str, saveTo: str, on_bytes=None):
    """Download to saveTo + ".part", resuming after dropped connections, and move it into place
    only when complete. Connections to the weights CDN were being cut every few seconds, which
    used to leave a truncated file at saveTo that was never fetched again."""
    dirname = os.path.dirname(saveTo)
    if dirname != "":
        os.makedirs(dirname, exist_ok=True)
    part = saveTo + ".part"
    total = None
    last_error = None
    stalled = 0
    started = time.time()
    prev_have = -1
    while True:
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if total is not None and have >= total:
            break
        stalled = 0 if have > prev_have else stalled + 1
        prev_have = have
        if stalled >= MAX_STALLED_TRIES or time.time() - started > MAX_SECONDS:
            raise RuntimeError(f"download failed, no progress after {stalled} tries: {url}: {last_error}")
        headers = {"Range": f"bytes={have}-"} if have > 0 else {}
        try:
            with requests.get(url, stream=True, allow_redirects=True, headers=headers, timeout=30) as req:
                if req.status_code == 416 and have > 0:  # nothing left to send
                    break
                if 400 <= req.status_code < 500:
                    # gated, removed or wrong URL: retrying won't help
                    raise PermissionError(f"HTTP {req.status_code} for {url}")
                req.raise_for_status()
                if have > 0 and req.status_code != 206:  # the server ignored the range: start over
                    have = 0
                length = req.headers.get("content-length")
                # a compressed response's length isn't the file's
                if length is not None and req.headers.get("content-encoding") is None:
                    total = have + int(length)
                with open(part, "ab" if have > 0 else "wb") as f:
                    for chunk in req.iter_content(chunk_size=CHUNK):
                        if chunk:
                            f.write(chunk)
                            if on_bytes is not None:
                                on_bytes(len(chunk), total)
            if total is None or os.path.getsize(part) >= total:
                break
        except PermissionError:
            raise
        except Exception as e:
            last_error = e
            logger.info(f"[Downloader] {os.path.basename(saveTo)}: connection dropped at {have} bytes ({e}), resuming")
            time.sleep(1.5)
    if total is not None and os.path.getsize(part) != total:
        raise RuntimeError(f"download incomplete: {url}: {os.path.getsize(part)} of {total} bytes")
    os.replace(part, saveTo)


def download(params):
    url = params["url"]
    saveTo = params["saveTo"]
    position = params["position"]
    progress_bar = tqdm(leave=False, unit="B", unit_scale=True, unit_divisor=1024, position=position)

    def on_bytes(n, total):
        if total is not None and progress_bar.total != total:
            progress_bar.total = total
        progress_bar.update(n)

    try:
        _fetch(url, saveTo, on_bytes)
    except Exception as e:
        # the file is left missing, so the caller sees the failure (and a later start retries)
        logger.warning(e)
    finally:
        progress_bar.close()


def download_no_tqdm(params):
    url = params["url"]
    saveTo = params["saveTo"]
    count = [0]

    def on_bytes(n, total):
        count[0] += 1
        if count[0] % 16 == 0:
            print(".", end="", flush=True)

    try:
        _fetch(url, saveTo, on_bytes)
        logger.info(f"[Voice Changer] download sample catalog. {saveTo}")
    except Exception as e:
        logger.warning(e)
