from dataclasses import dataclass


@dataclass
class DomainSpec:
    """Physical dimensions of the thin layer."""

    Lx: float
    Ly: float
    H: float


@dataclass
class LoadPatch:
    """Specification of the loaded patch on the top surface."""

    x_start: float
    x_end: float
    y_start: float
    y_end: float
    pressure: float


@dataclass
class Scaling:
    """Optional nondimensionalization scales."""

    U0: float = 1.0
    stress_scale: float = 1.0
