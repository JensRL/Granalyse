"""Granalyse: Granulation Analysis with Nested Sampling for Stellar Radius Inference."""

import jax

# Double precision throughout: the chi-squared likelihood and evidence
# integration are numerically sensitive.
jax.config.update("jax_enable_x64", True)

__version__ = "0.1.0"

from .models import BackgroundModel, available_models, get_model, register_model  # noqa: E402
from .star import FitCase, Star  # noqa: E402
