"""Windows PaddlePaddle 3.3.x CPU: disable broken OneDNN fused_conv2d path."""

from __future__ import annotations

import os

os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_onednn", "0")
os.environ.setdefault("FLAGS_USE_MKLDNN", "0")
os.environ.setdefault("FLAGS_ONEDNN", "0")
os.environ.setdefault("FLAGS_enable_mkldnn", "0")
os.environ.setdefault("FLAGS_use_dnnl", "0")

_PATCHED = False


def ensure_paddle_cpu_safe() -> None:
    global _PATCHED
    if _PATCHED:
        return

    import paddle.inference as inference

    original_config_init = inference.Config.__init__
    original_create_predictor = inference.create_predictor

    def patched_config_init(self, *args, **kwargs):
        original_config_init(self, *args, **kwargs)
        try:
            self.disable_mkldnn()
        except Exception:
            pass

    def patched_create_predictor(config):
        try:
            config.disable_mkldnn()
        except Exception:
            pass
        try:
            config.switch_ir_optim(False)
        except Exception:
            pass
        return original_create_predictor(config)

    inference.Config.__init__ = patched_config_init
    inference.create_predictor = patched_create_predictor
    _PATCHED = True
