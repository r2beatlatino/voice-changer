import numpy as np
import pytest
import torch

pytest.importorskip("df")


def test_model_is_loaded_once_and_skips_git(monkeypatch):
    import importlib

    import df.logger

    enhance_module = importlib.import_module("df.enhance")  # df.enhance is also a function
    import voice_changer.common.robust.DfnDenoiser as dfn_module

    loads = []

    class FakeState:
        def sr(self):
            return 48000

    def fake_init_df(**kwargs):
        loads.append(kwargs)
        return torch.nn.Linear(1, 1), FakeState(), None

    monkeypatch.setattr(enhance_module, "init_df", fake_init_df)
    monkeypatch.setattr(df.logger, "get_commit_hash", df.logger.get_commit_hash)
    monkeypatch.setattr(dfn_module, "_shared", {})
    a = dfn_module.DfnDenoiser()
    b = dfn_module.DfnDenoiser()
    assert len(loads) == 1
    assert a.model is b.model
    # each voice keeps its own streaming state
    a.push(np.ones(1600, dtype=np.float32))
    assert not np.array_equal(a.buf, b.buf)
    # no git on a user's PC: DeepFilterNet's logger must not shell out to it
    assert df.logger.get_commit_hash() is None
