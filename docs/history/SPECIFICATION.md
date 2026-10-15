# Frozen specification for Bitcoin synthetic evidence pilot

Version 0.1.0, October 12, 2026. This specification makes the first two rows of Table 3 executable and supplies data for the next row. Numerical supports are experimental choices, not estimates of Bitcoin investors. The original research design remains unchanged; this file records the implementation choices needed to execute it.

## Research target and current boundary

The planned paper asks which observations distinguish beliefs, financial resources, and institutional restrictions when similar observed behavior can imply different responses to funding. The population-level primary target is the mean paired price response, Ψ(θ), over independent populations and shocks at fixed environment θ. The current run provides pilot sample means and their Monte Carlo standard errors. Those means are not a sufficiently precise independent oracle for later coverage tests.

This release implements data generation, observation releases, and verification. It has no calibrated estimator, candidate confidence region, bootstrap comparison, omission experiment, or coverage result. The confirmation seed bank remains unused. The reduced three-environment run checks feasibility and variation, not empirical plausibility or the prevalence of ambiguity.

## Units and monthly sequence

Each economy starts with 2,000 investors, 100 units of cash, and 100 represented Bitcoin units. One cash unit per Bitcoin is the scaling reference; actual prices clear endogenously. These normalized units do not represent the outstanding Bitcoin stock or dollar prices. Individual arrays use floating-point arithmetic with 64-bit precision.

At each month t from 1 through 24:

1. Credit the declared external funding to each investor's cash account.
2. Update the common signal, subjective probabilities, and position ceilings for t.
3. Choose final holdings and solve for a common clearing price.
4. Set cash after trade to cash before trade plus price times Bitcoin sold. Record external flows and market residuals separately.
5. At months 6, 12, and 18, retain the post-trade accounts and the contemporaneous beliefs and ceilings. Surveys refer to these states.

Months 1–18 are the entire analyst information set. Months 19–24 are held out. Signal innovations are independent across markets. Control and treatment branches use identical realized signals and ordinary funding. Monthly funding is gF times initial individual cash, beginning in month 1; its aggregate is therefore 100gF per month. Existing cash earns no interest in this version.

Investors solve a rolling one-period portfolio problem and can rebalance without trading costs every month. The model does not impose rational expectations about its own price process. The two subjective values restrict the economic range of the model and do not support extrapolation to long-run dollar prices.

## Population distributions

Use NumPy's default generator with explicitly recorded `SeedSequence` streams. Each population begins with seven independent standard normal scores per investor. Zero positions, unrestricted mandates, and intervention membership use independent random draws from the same stream after the normal draws; their membership does not depend on hidden attributes.

| Quantity | Frozen construction | Support or meaning |
|---|---|---|
| Cash weight | exp[0.9 clip(Zcash, −2.5, 2.5)], then normalize across investors to 100 | Strictly positive, bounded raw weights |
| Holdings weight | exp[0.9 clip(0.4 Zcash + sqrt(1−0.4²) Zhold, −2.5, 2.5)] | Normalize remaining weights to 100 after assigning exactly 25% of investors zero holdings |
| Belief score | Zbelief = 0.6 Zcash + sqrt(1−0.6²) Zindependent | Gaussian latent correlation; not the realized correlation with usable cash |
| Initial high-state probability | q0 = 0.05 + 0.9 expit(0 + 0.8 Zbelief) | Between 0.05 and 0.95 |
| Risk coefficient | γ = 0.5 + 5.5 expit[logit((2−0.5)/5.5) + 0.8 Zrisk] | CRRA γ between 0.5 and 6; median 2 before sampling variation |
| Committed cash fraction | l = 0.9 expit[logit(0.55/0.9) + Zcommitment] | Between 0 and 0.9; median 0.55 |
| Required cash | L = l C0 | Fixed nominal amount throughout both branches |
| Initial allowance | A0 = C0 exp[log(0.25) + log(16) Φ(Zallowance)] | Between 0.25 and 4 times initial cash divided by the reference price |
| Position ceiling | Kt = H0 + A0 exp(gK t) | Bitcoin units; 20% of investors instead have no ceiling |
| Transfer group | Exactly 25% of investors, randomly selected before simulation | Membership independent of beliefs, resources, and constraints |

The population generator records realized belief correlations with both nominal and usable initial cash. Independent commitment fractions mean the latent belief–cash coefficient is not the realized belief–usable-cash coefficient. No post-generation adjustment forces the latter to a target. Risk, commitment, and allowance scores are otherwise independent in this version. The holdings and belief scores inherit dependence through cash.

## Beliefs and environments

All investors use two subjective future resale values, VL = 0.6 and VH = 1.8. Probability qt is clip{expit[logit(q0) + gB t + Bt], 0.05, 0.95}. The signal obeys Bt = 0.7 B(t−1) + εt, with B0 = 0 and εt independently normal with standard deviation 0.05. It is publicly observed. Record how often probability clipping occurs.

| Environment | Belief drift gB per month | Funding rate gF per month | Allowance growth gK per month |
|---|---:|---:|---:|
| Belief change | 0.02 | 0 | 0 |
| Additional funding | 0 | 0.01 | 0 |
| Mandate relaxation | 0 | 0 | 0.02 |

These are three separate fixed environments, with 50 independently seeded markets per environment. They are not selected by matching realized prices. The observation designs within each market are paired. The wider protocol permits gB in [−0.03, 0.03], gF in [0, 0.01], and gK in [−0.02, 0.02]; verification includes negative and mixed changes. All other nuisance parameters remain fixed for this first generation run. This is a deliberate reduction from the eventual estimation problem.

## Individual choice and market clearing

For cash C and holdings H entering trade at price P, define W = C + PH. Choose final holdings x to maximize

E[uγ(W + (Vs − P)x)]

subject to 0 ≤ x ≤ min[K, H + (C−L)/P]. The utility is log(w) when γ = 1 and (w^(1−γ)−1)/(1−γ) otherwise. Subtracting 1 changes only the utility origin. Terminal wealth must be strictly positive in both states. A negative feasible upper bound raises an economic infeasibility record; the code does not repair the budget or cash floor.

For VL < P < VH, define a = VH−P, b = P−VL, and s = expit{[logit(q) + log(a/b)]/γ}. The unconstrained optimizer is

x* = W(2s−1) / [a(1−s) + bs].

Clip this value to the feasible interval. At P ≤ VL the objective increases with holdings, so choose the upper bound. At P ≥ VH choose zero. The expression follows by equating the positive and negative state terms in the utility derivative. The verification suite independently maximizes the original expected utility and checks a representative-agent aggregate equilibrium with a closed-form price.

Clearing requires sum(x) = 100. Trades update C+ = C + P(H−x) and H+ = x. An increase in quoted Bitcoin value does not create cash. Search a grid with nine geometrically spaced points from 10^−8 to VL, 65 linearly spaced points from VL to VH, and the preceding price. Month 1 uses reference price 1 for root selection. Refine sign changes with Brent's method; inspect local minima of absolute excess demand for same-sign tangencies and retain contiguous sampled near-zero spans. Select the discovered root nearest the prior price, breaking exact ties downward.

The search is finite. The phrase 'one discovered root' does not mean a uniqueness theorem. A refinement check uses 257 central grid points with identical populations and shocks. Root counts and sampled spans remain in evaluation records. Prices outside the positive search floor are not resolved by this implementation. Core cash floors are initially feasible and ordinary funding is nonnegative.

## Binding constraints and numerical acceptance

An active constraint is a holdings bound reached within numerical tolerance. An economically binding upper bound also has a strictly positive utility derivative in the direction of more holdings. Normalize this derivative by the positive factor W^−γ before comparison. Record both cash and ceiling binding indicators; simultaneous binding is allowed and receives its own count. A binding indicator is not an additive causal attribution when restrictions interact.

Clearing residual tolerance is 10^−8 Bitcoin units. The cash ledger tolerance is 10^−7 cash units. Bound activity tolerance is 10^−8 times max(1, x, H); a positive normalized marginal must exceed 10^−8 for economic binding. KKT violation tolerance is 10^−6. Residuals remain in output rather than being eliminated by assigning a balancing trade to one investor. Failures retain the seed, environment, category, and error text.

## Paired intervention

After month 18, credit an external cash transfer equal to 5% of aggregate month-18 cash, distributed equally to the fixed random quarter. The treatment branch then receives ordinary month-19 funding and updates beliefs and ceilings exactly as the control branch does. The intervention does not change the schedule of commitments or mandate limits. Ordinary predeclared allowance growth continues in both branches; there is no treatment-specific limit relaxation.

The impact response is 100[P19(treatment)/P19(control)−1]. The secondary response uses month 24. Responses are percentages; their standard errors are percentage points. The transfer appears once in the external-flow ledger. A zero-transfer test must reproduce both branches exactly. A 10% transfer is included as a numerical stress case, not as a completed nonlinear-response study.

## Observation mappings

Draw a cohort of 250 respondents and retain it through all three survey dates. Clean participation has constant probability 0.5; sample 250 uniformly without replacement from respondents. Selective participation uses logistic odds increasing fourfold across one population interquartile increase in month-6 optimism. Solve the logistic intercept so mean participation probability is 0.5, then draw participation and sample respondents. The odds ratio describes participation, not final fixed-size inclusion odds. Insufficient respondents create a sampling failure, not a replacement market.

The selective regime changes participation only. Reports remain exact so that this failure can be isolated. Neither regime includes rounding, measurement error, stale mandates, or attrition after enrollment. Those are later observation experiments. Participation probabilities and population beliefs appear only in evaluation records.

| Panel | Released individual fields | Permitted links |
|---|---|---|
| O1 | Holdings and observable transfer-group membership | Same investor across waves |
| O2 | O1 plus subjective high-state probability | Holdings and beliefs jointly linked |
| O6 | O2 plus cash, commitment, ceiling or unrestricted status, and risk-task response | All modules linked by investor |
| O7 | Exactly the O6 marginal observations | Independent column permutations by module; links retained within each module across dates |

Every panel also receives prices and the common signal through month 18, initial aggregate cash and represented Bitcoin totals, survey dates, and its panel identifier. There are no other financial assets in the core model. O6 risk information is an idealized certainty equivalent for a 50–50 gamble paying 0.5 or 1.5 normalized wealth units under the investor's CRRA utility; the hidden γ itself is not released. This assumed task is not a validated survey instrument.

O7 has five modules: holdings; beliefs; resources (cash and commitments); permissions (ceilings and unrestricted status); and risk tasks. The transfer-group label stays attached to each module's records, allowing the same group marginals in O6 and O7. O7 column numbers cannot identify the same person across modules. No linked surrogate is constructed. Sorting checks confirm exactly identical marginal observations and group marginals across all O6/O7 pairs.

## Analysis interface and future inference

`features.summaries` uses only released fields. Initial candidates include selected price levels, survey quantiles, target-group means, and linked rank correlations where permitted. O7 does not expose cross-module correlations. These summaries are a prototype to be evaluated for information loss before fixing a discrepancy function; no covariance, threshold, confidence region, or fitted mechanism is supplied here.

The eventual estimator must vary nuisance parameters, particularly commitment levels and belief–resource dependence. Known fixed θ across three labels would make the problem artificially easy. Calibration must reproduce the sampling process or explicitly test its misspecification. An independent large simulation bank must establish Ψ(θ) and its Monte Carlo uncertainty before testing interval coverage. The ordinary market simulations in this release are neither that oracle bank nor the confirmation sample.

## Seeds and release records

Development seeds start at 110000, generation-pilot seeds at 220000, and reserved confirmation seeds at 330000. Add 10000 times the zero-based environment index and the replication index. Observation seeds add 1000000 to each market seed; the same observation seed supports clean/selective pairing. Root-refinement cases use 440000–440002; additional verification cases use 450000–450007. The runner refuses to label a run 'confirmation' because confirmation inference does not exist yet.

Each run retains its configuration and Python-source hashes before execution, dataset and market manifests, cash ledger, clearing diagnostics, error records, and software versions. Three full month-18 checkpoints support inspection; other markets reproduce from their evaluator-only seeds. Opaque dataset identifiers hide the generating environment and participation regime from the analyst files; panel identifiers remain visible because they define permitted joins. Reports may aggregate evaluation truth, but fitting code must never read it. Ordinary runtime differences are not failures of numerical reproduction.
