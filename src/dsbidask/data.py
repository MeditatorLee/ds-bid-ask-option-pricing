"""Load option quotes (Yahoo Finance CSV format) and build calibration inputs (thesis Chapter 5)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .config import S0_ASK, S0_BID

DATA_DIR = Path(__file__).resolve().parent / "datasets"
PathLike = Union[str, Path]


@dataclass
class Quotes:
    """Bid/ask quotes of calls and puts plus the underlying quote.

    The underlying is stored as a call with strike 0 (``K_call[0] == 0``), as in the thesis.
    """

    K_call: NDArray[np.float64]; C_bid: NDArray[np.float64]; C_ask: NDArray[np.float64]
    K_put: NDArray[np.float64]; P_bid: NDArray[np.float64]; P_ask: NDArray[np.float64]
    S0_bid: float = S0_BID
    S0_ask: float = S0_ASK


def load_side(path: PathLike, add_stock: bool = False, s0: Tuple[float, float] = (S0_BID, S0_ASK)
              ) -> Tuple[NDArray, NDArray, NDArray]:
    """Read ``strike, bid, ask`` from one CSV (sorted by strike); optionally prepend the stock as strike 0."""
    df = pd.read_csv(path)[["strike", "bid", "ask"]].astype(float).dropna().sort_values("strike")
    K, b, a = df.strike.to_numpy(), df.bid.to_numpy(), df.ask.to_numpy()
    if add_stock:
        K, b, a = np.r_[0.0, K], np.r_[s0[0], b], np.r_[s0[1], a]
    return K, b, a


def load_quotes(calls: Optional[PathLike] = None, puts: Optional[PathLike] = None, add_stock: bool = True,
                spot: Tuple[float, float] = (S0_BID, S0_ASK)) -> Quotes:
    """Load quotes; defaults to the thesis files (calls 42 + stock, puts 33; spread <= 1 USD).

    ``spot`` is the underlying ``(bid, ask)``, inserted as a strike-0 call when ``add_stock``.
    """
    calls = calls or DATA_DIR / "META_calls_2023-10-27_s_1_0_LOWER.csv"
    puts = puts or DATA_DIR / "META_puts_2023-10-27_s_1_0_LOWER.csv"
    Kc, Cb, Ca = load_side(calls, add_stock, spot)
    Kp, Pb, Pa = load_side(puts)
    return Quotes(Kc, Cb, Ca, Kp, Pb, Pa, spot[0], spot[1])
