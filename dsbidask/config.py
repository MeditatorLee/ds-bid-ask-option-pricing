"""Default market constants for the META 2023-09-29 -> 2023-10-27 case study."""
T_DEFAULT = 20                           # trading days to maturity
R_ANNUAL = 0.0555                        # annual risk-free rate
R_DEFAULT = (1 + R_ANNUAL) ** (1 / 250)  # one-period gross rate (250 trading days)
S0_BID, S0_ASK = 300.11, 300.30          # META quote on 2023-09-29
