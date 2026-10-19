"""
Per-voice settings survive a restart and don't leak between voices. Run from the server directory:
    python -m pytest tests -q
"""

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_changer.VoiceChangerManager import VoiceChangerManager, VoiceChangerManagerSettings  # noqa: E402


class _Recorder:
    def __init__(self):
        self.values = {}

    def update_settings(self, key, val):
        self.values[key] = val


def _manager(stored=None):
    m = VoiceChangerManager.__new__(VoiceChangerManager)
    m.settings = VoiceChangerManagerSettings()
    m.stored_setting = stored if stored is not None else {}
    m.serverDevice = _Recorder()
    m.voiceChanger = None

    def generate(slot):
        m.voiceChanger = _Recorder()  # a fresh model starts from its defaults

    m.generateVoiceChanger = generate
    m.get_info = lambda: {}
    return m


def test_voice_settings_are_kept_per_voice_and_restored(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m = _manager()
    m.update_settings("modelSlotIndex", 5)
    m.update_settings("tran", "12")
    m.update_settings("vadThreshold", "0.4")
    m.update_settings("extraConvertSize", "65536")
    m.update_settings("modelSlotIndex", 6)
    m.update_settings("denoiseStrong", "1")
    assert "tran" not in m.voiceChanger.values  # voice 5's pitch doesn't follow to voice 6

    # restart: a new manager reads the file and loads the last voice used
    m2 = _manager(json.load(open(tmp_path / "stored_setting.json")))
    m2.update_settings("modelSlotIndex", 5)
    assert m2.voiceChanger.values["tran"] == "12"
    assert m2.voiceChanger.values["vadThreshold"] == "0.4"
    assert m2.voiceChanger.values["extraConvertSize"] == "65536"
    assert "denoiseStrong" not in m2.voiceChanger.values
    m2.update_settings("modelSlotIndex", 6)
    assert m2.voiceChanger.values["denoiseStrong"] == "1"


def test_a_new_model_in_the_slot_forgets_the_old_settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m = _manager()
    m.update_settings("modelSlotIndex", 5)
    m.update_settings("tran", "12")
    m._forget_voice_settings(5)
    assert json.load(open(tmp_path / "stored_setting.json"))["voices"] == {}
