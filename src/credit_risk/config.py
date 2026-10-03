"""Central configuration. Every business assumption lives here so it can be challenged in one place."""
from dataclasses import dataclass, field

RANDOM_STATE = 42

# Columns kept from application_train.csv. Protected attributes (gender, family status)
# are deliberately excluded so the score cannot use them directly.
APPLICATION_NUMERIC = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "REGION_RATING_CLIENT",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
]
APPLICATION_CATEGORICAL = ["NAME_CONTRACT_TYPE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE"]
TARGET = "TARGET"
ID = "SK_ID_CURR"


@dataclass(frozen=True)
class PricingAssumptions:
    """Annualised inputs for risk based pricing."""
    cost_of_funds: float = 0.06       # lender's funding cost
    operating_cost: float = 0.04      # servicing cost for small ticket loans
    target_margin: float = 0.03       # required return on top of costs
    lgd: float = 0.65                 # loss given default for unsecured MSME credit
    max_apr: float = 0.36             # responsible lending cap, never price above this


@dataclass(frozen=True)
class TierPolicy:
    name: str
    max_pd: float              # upper PD bound for the tier
    limit_income_multiple: float  # credit limit as a multiple of monthly income
    max_tenure_months: int


# Five tiers from lowest to highest risk. PD ceilings are recalibrated on the
# validation set in tiers.calibrate_tiers; these values are only fallbacks.
DEFAULT_TIERS = [
    TierPolicy("A", 0.03, 3.0, 24),
    TierPolicy("B", 0.06, 2.0, 18),
    TierPolicy("C", 0.10, 1.5, 12),
    TierPolicy("D", 0.18, 1.0, 6),
    TierPolicy("E", 1.00, 0.0, 0),   # decline: refer to coaching and bookkeeping first
]

TIER_QUANTILES = [0.30, 0.55, 0.75, 0.90]   # share of applicants at or below each tier ceiling


@dataclass
class Config:
    pricing: PricingAssumptions = field(default_factory=PricingAssumptions)
    tiers: list = field(default_factory=lambda: list(DEFAULT_TIERS))
    test_size: float = 0.2
    valid_size: float = 0.2
    max_expected_loss_rate: float = 0.03   # risk appetite: expected loss as a share of exposure
