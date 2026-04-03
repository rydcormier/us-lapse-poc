from __future__ import annotations

# Censorship date for incomplete policies
CENSOR_DATE = "2008-12-31"

# Modeling table columns (built in build_features.py)
CAT_COLS = [
    "gender",
    "premium_frequency",
    "risk_state",
    "underwriting_age",
    "living_place",
    "acc_death_rider",
]

NUM_COLS = [
    "tenure_qtr",
    "annual_premium",
]

# Optional numeric macro (disabled by default)
DJIA_COL = "djia"
