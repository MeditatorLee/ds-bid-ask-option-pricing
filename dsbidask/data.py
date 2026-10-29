"""Load option quotes (Yahoo Finance CSV format) and build calibration inputs."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from .config import S0_BID, S0_ASK

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Quotes:
    K_call: np.ndarray; C_bid: np.ndarray; C_ask: np.ndarray
    K_put: np.ndarray;  P_bid: np.ndarray; P_ask: np.ndarray
    S0_bid: float = S0_BID; S0_ask: float = S0_ASK


def load_side(path, add_stock: bool = False, s0=(S0_BID, S0_ASK)):
    df = pd.read_csv(path)[["strike", "bid", "ask"]].astype(float).dropna()
    df = df.sort_values("strike")
    K, b, a = df.strike.to_numpy(), df.bid.to_numpy(), df.ask.to_numpy()
    if add_stock:  # the stock itself is a strike-0 call
        K, b, a = np.r_[0.0, K], np.r_[s0[0], b], np.r_[s0[1], a]
    return K, b, a


def load_quotes(calls=None, puts=None, add_stock=True) -> Quotes:
    calls = calls or DATA_DIR / "META_calls_2023-10-27_s_1_0_LOWER.csv"
    puts = puts or DATA_DIR / "META_puts_2023-10-27_s_1_0_LOWER.csv"
    Kc, Cb, Ca = load_side(calls, add_stock)
    Kp, Pb, Pa = load_side(puts)
    return Quotes(Kc, Cb, Ca, Kp, Pb, Pa)
