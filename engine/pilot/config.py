"""Frozen experimental settings; no empirical calibration is implied."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Config:
    n: int = 2000
    periods: int = 24
    cutoff: int = 18
    survey_months: tuple = (6, 12, 18)
    sample_size: int = 250
    initial_cash: float = 100.0
    stock: float = 100.0
    value_low: float = 0.6
    value_high: float = 1.8
    belief_location: float = 0.0
    belief_sd: float = 0.8
    risk_median: float = 2.0
    risk_latent_sd: float = 0.8
    wealth_dispersion: float = 0.9
    belief_cash_correlation: float = 0.6
    cash_holdings_correlation: float = 0.4
    commitment_median: float = 0.55
    commitment_latent_sd: float = 1.0
    zero_holdings_share: float = 0.25
    unrestricted_share: float = 0.2
    belief_drift: float = 0.0
    funding_rate: float = 0.0
    allowance_growth: float = 0.0
    signal_ar: float = 0.7
    signal_innovation_sd: float = 0.05
    transfer_share: float = 0.05
    target_share: float = 0.25
    selection_odds_ratio: float = 4.0
    participation_mean: float = 0.5
    root_grid: int = 65
    root_floor: float = 1e-8
    root_xtol: float = 1e-12
    clearing_tolerance: float = 1e-8
    accounting_tolerance: float = 1e-7

    def validate(self):
        if self.n < 4 or self.sample_size > self.n or self.sample_size < 2:
            raise ValueError("Invalid population or sample size")
        if not 0 < self.cutoff < self.periods:
            raise ValueError("Intervention must precede the final month")
        if any(t < 1 or t > self.cutoff for t in self.survey_months):
            raise ValueError("Survey after information cutoff")
        if not 0 < self.value_low < self.value_high:
            raise ValueError("Invalid subjective value support")
        if not 0 < self.initial_cash or not 0 < self.stock:
            raise ValueError("Positive aggregate endowments required")
        if not 0 < self.commitment_median < .9 or not .5 < self.risk_median < 6:
            raise ValueError("Distribution median outside support")
        if not -.03 <= self.belief_drift <= .03:
            raise ValueError("Belief drift outside protocol support")
        if not 0 <= self.funding_rate <= .01 or not -.02 <= self.allowance_growth <= .02:
            raise ValueError("Funding or allowance drift outside support")
        if not -1 < self.belief_cash_correlation < 1:
            raise ValueError("Invalid Gaussian dependence")
        if not -1 < self.cash_holdings_correlation < 1:
            raise ValueError("Invalid holdings dependence")
        if not 0 <= self.transfer_share or not 0 < self.target_share < 1:
            raise ValueError("Invalid intervention")
        if self.selection_odds_ratio <= 0 or not 0 < self.participation_mean < 1:
            raise ValueError("Invalid participation mechanism")
        if self.root_grid < 9 or self.root_floor <= 0:
            raise ValueError("Invalid root search")
        for k,v in asdict(self).items():
            if isinstance(v,float) and not math.isfinite(v):
                raise ValueError(f"Nonfinite configuration {k}")
        return self


ENVIRONMENTS = {
    "belief": dict(belief_drift=0.02),
    "funding": dict(funding_rate=0.01),
    "mandate": dict(allowance_growth=0.02),
}
