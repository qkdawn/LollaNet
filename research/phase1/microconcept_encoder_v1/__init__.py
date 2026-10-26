"""Phase-1 microconcept state encoder v1.

This package intentionally consumes only q/g, point-in-time factor values, and
structured MD&A claim metadata.  It never reads routing labels, future outcomes,
issuer identity as a feature, raw MD&A text, LLM hidden states, or steering data.
"""
from .data import EncoderData, build_encoder_data, make_splits
from .model import DualHeadMicroconceptEncoder

__all__ = ["EncoderData", "DualHeadMicroconceptEncoder", "build_encoder_data", "make_splits"]
