"""ARGUS: real-time seismic source characterization for sparse-network EGS/CCS.

Reference implementation accompanying:
  Kurosawa, I. ARGUS: A 17-ms End-to-End Deep Learning Pipeline for Real-Time
  Seismic Source Characterization and Ground Motion Prediction in Sparse-Network
  EGS/CCS Environments. Seismological Research Letters (submitted, 2026).
"""
from .pipeline import ARGUS, resolve_device

__version__ = "1.0.0"
__all__ = ["ARGUS", "resolve_device"]
