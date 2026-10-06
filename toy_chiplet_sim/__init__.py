"""
Multi-chiplet GPU simulator for MoE (Mixture-of-Experts) model execution.

This package provides cycle-accurate simulation of MoE inference on a 2D mesh
of chiplets with configurable allocation and execution strategies.
"""

from .config import SystemConfig, TimingConfig
from .simulation.simulator import Simulator

__all__ = ["SystemConfig", "TimingConfig", "Simulator"]
