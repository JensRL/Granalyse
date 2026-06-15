"""Model registry: look up background models by letter or alias, and
register user-defined models."""

from .base import BackgroundModel
from .builtin import MODEL_D, MODEL_H, MODEL_J, MODEL_T

_REGISTRY = {}


def register_model(model: BackgroundModel):
    """Register a BackgroundModel under its name and aliases."""
    for key in (model.name, *model.aliases):
        key = key.lower()
        if key in _REGISTRY and _REGISTRY[key] is not model:
            raise ValueError(f"Model name/alias '{key}' is already registered.")
        _REGISTRY[key] = model


def get_model(name) -> BackgroundModel:
    """Return the model registered under `name` (letter or alias, case-insensitive)."""
    if isinstance(name, BackgroundModel):
        return name
    try:
        return _REGISTRY[name.lower()]
    except KeyError:
        known = sorted({m.name for m in _REGISTRY.values()})
        raise KeyError(f"Unknown background model '{name}'. Known models: {known}") from None


def available_models():
    """Unique registered models, keyed by canonical name."""
    return {m.name: m for m in _REGISTRY.values()}


for _m in (MODEL_D, MODEL_J, MODEL_H, MODEL_T):
    register_model(_m)
