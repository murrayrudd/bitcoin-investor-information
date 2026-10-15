"""Two-state CRRA choice, numerical clearing, and paired counterfactuals.

The individual demand formula is analytic. Aggregate clearing is searched
numerically, with a finite-grid diagnostic for additional roots. No aggregate
uniqueness theorem is assumed.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit, logit, ndtr
from scipy.optimize import brentq, minimize_scalar
from .config import Config
from .kernel import excess_function


class EconomicInfeasibility(RuntimeError):
    pass


class NumericalFailure(RuntimeError):
    pass


@dataclass
class Population:
    cash0: np.ndarray
    holdings0: np.ndarray
    floor: np.ndarray
    gamma: np.ndarray
    belief_intercept: np.ndarray
    allowance0: np.ndarray
    unrestricted: np.ndarray
    target: np.ndarray


@dataclass
class State:
    month: int
    price: float
    cash: np.ndarray
    holdings: np.ndarray
    belief: np.ndarray
    cap: np.ndarray


@dataclass
class Market:
    population: Population
    signal: np.ndarray
    prices: np.ndarray
    snapshots: dict
    control_final: State
    treatment_final: State
    response: float
    response_six: float
    transfer: float
    ledger: list
    diagnostics: list


def population(cfg: Config, rng: np.random.Generator) -> Population:
    cfg.validate()
    z = rng.normal(size=(cfg.n, 7))
    cash_z = z[:, 0]
    rho = cfg.belief_cash_correlation
    belief_z = rho * cash_z + np.sqrt(1-rho*rho) * z[:, 1]
    rh = cfg.cash_holdings_correlation
    holdings_z = rh * cash_z + np.sqrt(1-rh*rh) * z[:, 2]
    # Winsorized normal scores give bounded positive wealth weights.
    c = np.exp(cfg.wealth_dispersion * np.clip(cash_z, -2.5, 2.5))
    h = np.exp(cfg.wealth_dispersion * np.clip(holdings_z, -2.5, 2.5))
    h[rng.choice(cfg.n, round(cfg.zero_holdings_share*cfg.n), replace=False)] = 0
    c *= cfg.initial_cash/c.sum()
    h *= cfg.stock/h.sum()
    q0 = .05 + .90*expit(cfg.belief_location + cfg.belief_sd*belief_z)
    risk_loc = logit((cfg.risk_median-.5)/5.5)
    gamma = .5 + 5.5*expit(risk_loc + cfg.risk_latent_sd*z[:, 3])
    commitment = .9*expit(logit(cfg.commitment_median/.9) + cfg.commitment_latent_sd*z[:, 4])
    allowance = c*np.exp(np.log(.25)+np.log(16)*ndtr(z[:, 5]))
    unrestricted = np.zeros(cfg.n, bool)
    unrestricted[rng.choice(cfg.n, round(cfg.unrestricted_share*cfg.n), replace=False)] = True
    target = np.zeros(cfg.n, bool)
    target[rng.choice(cfg.n, round(cfg.target_share*cfg.n), replace=False)] = True
    return Population(c, h, c*commitment, gamma, logit(q0), allowance, unrestricted, target)


def ordinary_inputs(pop, cfg, month, signal):
    raw = expit(pop.belief_intercept + cfg.belief_drift*month + signal)
    q = np.clip(raw, .05, .95)
    cap = pop.holdings0 + pop.allowance0*np.exp(cfg.allowance_growth*month)
    cap = np.where(pop.unrestricted, np.inf, cap)
    return q, cap, float(np.mean(q != raw))


def demand(prices, cash, holdings, floor, cap, q, gamma, low=.6, high=1.8):
    """Final Bitcoin holdings; scalar price -> N, vector prices -> P by N."""
    scalar = np.ndim(prices) == 0
    p = np.atleast_1d(prices).astype(float)[:, None]
    if np.any(p <= 0):
        raise ValueError("Prices must be positive")
    wealth = cash[None, :] + p*holdings[None, :]
    upper = np.minimum(cap[None, :], holdings[None, :] + (cash-floor)[None, :]/p)
    if np.any(wealth <= 0) or np.any(upper < -1e-10):
        raise EconomicInfeasibility("Nonpositive wealth or infeasible cash floor")
    upper = np.maximum(upper, 0)
    x = np.zeros_like(upper)
    below = p[:, 0] <= low
    x[below] = upper[below]
    interior = (p[:, 0] > low) & (p[:, 0] < high)
    if np.any(interior):
        pi = p[interior]
        a, b = high-pi, pi-low
        s = expit((logit(q)[None, :] + np.log(a)-np.log(b))/gamma[None, :])
        unconstrained = wealth[interior]*(2*s-1)/(a*(1-s)+b*s)
        x[interior] = np.clip(unconstrained, 0, upper[interior])
    return x[0] if scalar else x


def marginal_scaled(x, price, cash, holdings, q, gamma, low=.6, high=1.8):
    wealth = cash+price*holdings
    wl = (wealth+(low-price)*x)/wealth
    wh = (wealth+(high-price)*x)/wealth
    if np.any(wl <= 0) or np.any(wh <= 0):
        raise EconomicInfeasibility("Nonpositive terminal state wealth")
    return (1-q)*(low-price)*wl**(-gamma)+q*(high-price)*wh**(-gamma)


def discover_roots(fun, grid, previous, tolerance, xtol=1e-12):
    """Finite search, including sampled zero spans and same-sign tangencies.

    A detected zero span is only the sampled span, not a proof of its complete
    extent. Narrow roots can escape any finite scan; refinement is a diagnostic.
    """
    grid = np.unique(np.sort(np.append(grid, previous)))
    vals = np.asarray([fun(float(p)) for p in grid])
    if np.any(~np.isfinite(vals)):
        raise NumericalFailure("Nonfinite excess demand")
    zeros = np.abs(vals) <= tolerance
    roots, spans = [], []
    i = 0
    while i < len(grid):
        if zeros[i]:
            j = i
            while j+1 < len(grid) and zeros[j+1]:
                j += 1
            if j > i:
                spans.append([float(grid[i]), float(grid[j])])
                roots.append(float(np.clip(previous, grid[i], grid[j])))
            else:
                roots.append(float(grid[i]))
            i = j+1
        else:
            i += 1
    for i in range(len(grid)-1):
        if not zeros[i] and not zeros[i+1] and vals[i]*vals[i+1] < 0:
            roots.append(float(brentq(fun, grid[i], grid[i+1], xtol=xtol, rtol=1e-14)))
    for i in range(1, len(grid)-1):
        if zeros[i-1:i+2].any():
            continue
        if vals[i-1]*vals[i+1] > 0 and abs(vals[i]) < min(abs(vals[i-1]), abs(vals[i+1])):
            opt = minimize_scalar(lambda p: abs(fun(p)), bounds=(grid[i-1],grid[i+1]),
                                  method="bounded", options={"xatol":xtol})
            if opt.success and abs(fun(opt.x)) <= tolerance:
                roots.append(float(opt.x))
    roots = sorted(roots)
    unique = []
    for r in roots:
        if not unique or abs(r-unique[-1]) > 1e-8:
            unique.append(r)
    if not unique:
        raise NumericalFailure("No clearing root found in finite search domain")
    selected = min(unique, key=lambda p:(abs(p-previous), p))
    return selected, unique, spans


def clearing(cash, holdings, pop, q, cap, previous, cfg):
    stock = cfg.stock
    def excess(p):
        return float(demand(p,cash,holdings,pop.floor,cap,q,pop.gamma,
                            cfg.value_low,cfg.value_high).sum()-stock)
    accelerated=excess_function(cash,holdings,pop.floor,cap,q,pop.gamma,
                               cfg.value_low,cfg.value_high,stock)
    if accelerated is not None:excess=accelerated
    grid = np.unique(np.r_[np.geomspace(cfg.root_floor,cfg.value_low,9),
                           np.linspace(cfg.value_low,cfg.value_high,cfg.root_grid)])
    p, roots, spans = discover_roots(excess,grid,previous,cfg.clearing_tolerance,cfg.root_xtol)
    x = demand(p,cash,holdings,pop.floor,cap,q,pop.gamma,cfg.value_low,cfg.value_high)
    c = cash+p*(holdings-x)
    residual = float(x.sum()-stock)
    if abs(residual)>cfg.clearing_tolerance or c.min() < -cfg.accounting_tolerance:
        raise NumericalFailure("Clearing residual or negative cash")
    if np.min(c-pop.floor) < -cfg.accounting_tolerance:
        raise NumericalFailure("Cash commitment violated")
    if np.max(x-cap)>cfg.accounting_tolerance:
        raise NumericalFailure("Position ceiling violated")
    g = marginal_scaled(x,p,cash,holdings,q,pop.gamma,cfg.value_low,cfg.value_high)
    budget_upper = holdings+(cash-pop.floor)/p
    tol = 1e-8*np.maximum(1,np.maximum(x,holdings))
    cash_active = abs(x-budget_upper) <= tol
    cap_active = np.isfinite(cap) & (abs(x-cap)<=tol)
    positive = g > 1e-8
    lower = x <= tol
    interior = ~lower & ~cash_active & ~cap_active
    movable = np.minimum(budget_upper,cap)>tol
    kkt = max(float(np.max(np.maximum(g[lower&movable],0),initial=0)),
              float(np.max(np.maximum(-g[(cash_active|cap_active)&movable],0),initial=0)),
              float(np.max(abs(g[interior]),initial=0)))
    diag = dict(price=p, roots=roots, zero_spans=spans,
                clearing_residual=residual, cash_residual=float(c.sum()-cash.sum()),
                kkt_violation=kkt, cash_active=float(cash_active.mean()),
                cap_active=float(cap_active.mean()),
                cash_binding=float((cash_active&positive).mean()),
                cap_binding=float((cap_active&positive).mean()),
                joint_binding=float((cash_active&cap_active&positive).mean()),
                zero_position=float(lower.mean()))
    if kkt > 1e-6:
        raise NumericalFailure(f"Choice KKT violation {kkt}")
    return p,c,x,diag


def simulate(cfg: Config, seed: int) -> Market:
    cfg.validate()
    streams = np.random.SeedSequence(seed).spawn(2)
    pop = population(cfg,np.random.default_rng(streams[0]))
    rng = np.random.default_rng(streams[1])
    signal = np.empty(cfg.periods)
    previous = 0.0
    for t in range(cfg.periods):
        signal[t]=cfg.signal_ar*previous+rng.normal(0,cfg.signal_innovation_sd)
        previous=signal[t]
    c,h,p=pop.cash0.copy(),pop.holdings0.copy(),1.0
    prices, snapshots, ledger, diagnostics=[],{},[],[]
    monthly_funding=cfg.funding_rate*pop.cash0
    def step(c,h,p,t,branch,extra=0.0):
        # Treatment's one-time transfer is credited after month 18, before
        # month 19 ordinary funding and clearing; it is not repeated later.
        before=float(c.sum())
        incoming=monthly_funding.copy()
        if extra:
            incoming[pop.target] += extra/pop.target.sum()
        c=c+incoming
        q,cap,clipped=ordinary_inputs(pop,cfg,t,signal[t-1])
        p,c,h,d=clearing(c,h,pop,q,cap,p,cfg)
        d.update(month=t,branch=branch,belief_clipped=clipped)
        diagnostics.append(d)
        row=dict(month=t,branch=branch,cash_before=before,
                 ordinary_funding=float(monthly_funding.sum()),transfer=float(extra),
                 cash_after=float(c.sum()),bitcoin_after=float(h.sum()))
        row["cash_ledger_residual"]=row["cash_after"]-before-row["ordinary_funding"]-extra
        if abs(row["cash_ledger_residual"])>cfg.accounting_tolerance:
            raise NumericalFailure("External funding ledger does not reconcile")
        ledger.append(row)
        return State(t,p,c,h,q,cap)
    for t in range(1,cfg.periods+1):
        s=step(c,h,p,t,"control")
        c,h,p=s.cash,s.holdings,s.price
        prices.append(p)
        if t in set(cfg.survey_months)|{cfg.cutoff}:
            snapshots[t]=s
    control_final=s
    base=snapshots[cfg.cutoff]
    transfer=cfg.transfer_share*float(base.cash.sum())
    c,h,p=base.cash.copy(),base.holdings.copy(),base.price
    impact=None
    for t in range(cfg.cutoff+1,cfg.periods+1):
        s=step(c,h,p,t,"treatment",transfer if t==cfg.cutoff+1 else 0.0)
        c,h,p=s.cash,s.holdings,s.price
        if t==cfg.cutoff+1:
            impact=100*(p/prices[t-1]-1)
    return Market(pop,signal,np.asarray(prices),snapshots,control_final,s,
                  float(impact),float(100*(p/prices[-1]-1)),transfer,ledger,diagnostics)
