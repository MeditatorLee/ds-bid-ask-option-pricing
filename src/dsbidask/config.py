"""Default market constants for the META case study (thesis Chapter 5)."""
from __future__ import annotations

T_DEFAULT: int = 20
"""Trading days between 2023-09-29 and 2023-10-27."""

R_ANNUAL: float = 0.0555
"""Annual risk-free rate (1-month US Treasury bill)."""

R_DEFAULT: float = (1 + R_ANNUAL) ** (1 / 250)
"""One-period gross rate ``R = 1 + r``, 250 trading days per year (thesis: ``1 + r = 1.0555**(1/250)``)."""

S0_BID: float = 300.11
S0_ASK: float = 300.30
"""META stock quote on 2023-09-29; inserted as a call with strike 0."""
