"""Lazy camera exports; configuration imports must not initialize USD before Kit."""

from importlib import import_module

__all__ = ["GaussianSplatCamera", "GaussianSplatCameraCfg"]


def __getattr__(name):
    modules = {"GaussianSplatCamera": ".gaussian_splat_camera", "GaussianSplatCameraCfg": ".gaussian_splat_camera_cfg"}
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
