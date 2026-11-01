"""Bid-ask option pricing under Dempster-Shafer uncertainty (binomial DS multiplicative random walk)."""
from .config import R_DEFAULT, T_DEFAULT, S0_BID, S0_ASK
from .pricing import (b_up, european_bid_ask, european_call_bid_ask, european_put_bid_ask,
                      european_calls_bid_ask, european_puts_bid_ask, european_tree, crr_price)
from .choquet import bid_ask, bid_ask_trees, american_put_bid_ask, early_exercise_nodes
from .replication import replicating_portfolio, is_self_financing

__version__ = "0.3.0"
__all__ = [
    "R_DEFAULT", "T_DEFAULT", "S0_BID", "S0_ASK", "b_up",
    "european_bid_ask", "european_call_bid_ask", "european_put_bid_ask",
    "european_calls_bid_ask", "european_puts_bid_ask", "european_tree", "crr_price",
    "bid_ask", "bid_ask_trees", "american_put_bid_ask", "early_exercise_nodes",
    "replicating_portfolio", "is_self_financing",
]
