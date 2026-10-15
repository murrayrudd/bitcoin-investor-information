# Bitcoin synthetic evidence inference repair

Version 0.4.0. October 2026.

## Purpose and decision

This revision tests whether better simulator support and parameter-dependent calibration repair the failures documented in version 0.3.0. It keeps the economic model, nine-dimensional parameter domain, intervention, observation summaries, and scientific target unchanged. The target remains the population-average month-19 percentage price response to a transfer equal to 5% of month-18 aggregate cash, allocated to a fixed random quarter of the population. Analysts observe only months 1–18.

The full confirmation experiment is not part of this repair pass. The earlier confirmation seeds remain reserved. Read the revised results report before deciding whether to proceed. A finite collection of successful coverage checks does not establish coverage at every point in the continuous domain.

## Development information and simulation allocation

All version 0.3 findings are now development information, including its boundary failures and selected endpoints. The old 12 settings can therefore support regression checks but are not unseen settings. New independent markets at those settings test repeatability after repair. Twelve additional locations, fixed before their markets were generated, test transfer to previously unused settings.

Two fitted banks use disjoint market and observation seeds. Each reuses its own historical training and check simulations. Historical response-reference simulations supply additional response-regression training data; they do not calibrate the revised error allowance.

Each bank adds 812 training locations: 512 corner-shell locations with eight markets each, 192 face locations with 12 markets each, 96 locations from earlier optimized endpoints with 16 markets each, and the 12 earlier case locations with 32 markets each. Endpoint support uses a deterministic farthest-point allocation among previously reported fits. It does not use any revised validation result.

The three fresh design components are:

| Component | Parameter allocation |
|---|---|
| Bulk | The earlier mixture: probability 3/4 on the uniform mixed box and 1/12 on each pure-mechanism subspace. |
| Faces | Start with a uniform box draw; move one to four randomly selected coordinates to signed magnitudes between 0.8 and 1.0. |
| Corners | Every normalized coordinate has a random sign and a magnitude between 0.8 and 1.0. |

These are experimental allocation rules, not estimates of investor distributions. Components overlap; they are not disjoint partitions of the domain. Participation is represented on a logarithmic odds scale as in version 0.3.

Each bank also generates 96 check locations with 16 markets each, 384 quantile-training locations with 24 markets each, 599 calibration locations per component with one market each, and 99 response-reference locations per component with 32 markets each. Check and quantile locations allocate equal counts across the three components. All five fresh roles have separate seed namespaces.

## Mean and response approximations

Candidate forward approximations are cubic and quartic polynomials in normalized parameters. Weighted ridge regression uses setting means, weights proportional to the number of simulated markets, a penalty of 0.1, and an unpenalized intercept. Expected public-signal summaries remain zero. The check bank selects one common degree across the four observation panels by their average standardized mean-square error. This selection is development, not a validation result.

Response regression makes its own cubic-versus-quartic choice using the same check locations. It excludes every term containing participation strength: observation participation cannot change the population response. The pooled within-setting covariance retains the earlier 10% diagonal shrinkage. The revision changes the forward approximation and cutoff, not the economic engine or summary definitions.

## Parameter-dependent observation cutoff

For panel p, let S_p(theta, y) denote squared Mahalanobis discrepancy between released summaries y and their emulated mean. Independent quantile-training simulations fit the 96.5th conditional quantile of log S as a quadratic polynomial in theta. The implementation minimizes a smoothed quantile loss with smoothing parameter 0.03 and ridge penalty 0.002 on nonintercept coefficients. Each location and each replication receive equal weight within this bank.

The fitted function produces a positive preliminary cutoff q_p(theta). At each independent calibration location, form S_p(theta, Y)/q_p(theta). Within each of the three allocation components, select order statistic ceil(600 times 0.965) from 599 ratios. Multiply q_p(theta) by the largest of those three corrections. The same smooth function then applies throughout the domain, avoiding discontinuous changes at artificial regional boundaries.

Conditional on development and quantile fitting, rank calibration protects observation acceptance marginally over each declared component's random parameter allocation. Taking the largest correction cannot weaken any component's marginal protection. It does not prove acceptance at 96.5% for every fixed theta. Approximation quality of the conditional cutoff is assessed using independent fixed-case and selected-endpoint diagnostics.

## Response allowance

At each fresh response-reference location, add the two-sided 99.5% Student-t half-width for the simulated response mean to the absolute difference between that mean and the response approximation. For each allocation component, take order statistic ceil(100 times 0.99) from 99 values: its maximum. Apply the largest of the three component allowances to every projected interval.

This is deliberately more protective of boundaries than the earlier bulk-only allowance. Student-t coverage for the simulator mean remains approximate. Observation rejection of 3.5%, response rank error of 1%, and reference uncertainty of 0.5% give an approximate 95% component-marginal target before numerical search error. They do not supply a uniform response approximation bound or a guarantee at optimizer-selected endpoints. Fresh direct endpoint audits address that distinction empirically.

## Inversion and numerical checks

For each isolated release, the fitter searches the full continuous nine-dimensional box. Feasibility is S_p(theta, y)/c_p(theta) at most one. Analytic derivatives include the parameter-dependent cutoff. Every start first minimizes this ratio with bounded L-BFGS-B and then searches both response endpoints with constrained SLSQP. A feasible candidate remains useful even if an optimizer reports nonconvergence; all such statuses are counted.

Starting values alternate proximity to the released observations with separation in parameter space. The pool contains 1,024 scrambled Sobol locations and all forward-training locations. Main search budgets are 16 starts for holdings alone and 12 for the other panels. Four-start results are retained. The first three markets per setting also receive 48 starts for holdings alone or 32 for the other panels. Larger budgets retain earlier feasible endpoints. Movements above 0.005 response percentage points are flagged, without treating that threshold as an error bound or a certificate of global optimality.

The isolated-release CLI reads only a portable fitted model and a protected observation release. Generating parameters, evaluation seeds, future outcomes, and response references are evaluator-only. Known participation enters only a separately labeled sensitivity analysis.

## Frozen validation

After fitting, the pipeline hashes all inferential source and fitted models before generating validation markets. There are 24 fixed settings: 12 previously studied and 12 new. Each receives 80 evaluation markets and a separate 512-market response reference. Both fitted banks evaluate the same 1,920 markets. They are paired comparisons, not 3,840 independent markets.

Every setting reports true-parameter acceptance, response coverage, empty-set frequency, interval width, and Wilson intervals for the proportions. Empty sets count as misses and have no reported width. Response coverage is also checked against the independent oracle's 99% interval, distinguishing definite from possible containment. Pooled results describe this selected collection, not a representative sample of real Bitcoin markets or the entire parameter domain.

The first eight markets in each setting receive old version 0.3 fits for a paired method comparison. The same subset receives an O6 known-participation benchmark; its cutoff is unchanged and it has no separate conditional-coverage claim. Linked-versus-unlinked comparisons retain identical marginal observations and use paired differences in width and coverage. Monte Carlo standard errors condition on each fitted bank and the fixed case collection.

## Fresh selected-endpoint audit

Bank 0 O6 contributes both endpoints of its first release in all 24 settings. Settings 0, 9, 10, 12, 16, and 20 additionally contribute both endpoints for bank 0 O7 and bank 1 O6/O7. Each selected location receives 128 fresh markets, for at most 84 endpoints. Empty fits are reported and supply no endpoints. Selection follows a fixed rule; these audits do not tune the frozen models or their reported intervals.

The audit measures response approximation error and true-parameter acceptance at optimized locations. It reports containment of both independently estimated means and their 99% reference intervals, including failures. The simulator also rechecks market clearing, cash accounting, first-order optimality, finite root scans, release reconstruction, identical O6/O7 marginal summaries, and separation of seeds and evaluator information.

## Interpretation and continuation

Useful improvement requires assessing coverage and precision together. A positive linkage gain is not a success requirement. If recalibration removes an apparent gain, that result matters for the paper. Passing tests of numerical code is separate from achieving adequate statistical coverage.

The next scientific stage, once a defensible baseline exists, challenges the fitted model with omitted mechanisms and observation failures. This revision does not implement those misspecification experiments or narrative diffusion. It establishes whether the intended inference is dependable enough under its own assumptions to make those later comparisons informative.

## Additional numerical work after validation

The frozen 16-to-48-start check found three material holdings-only endpoint movements in bank 0. A recorded experiment reflected endpoint coordinates and restarted optimization, but comparison with 96 ordinary starts exposed further missed solutions. The package retains that experiment in `refined_O1/` for provenance; it is not the final holdings-only benchmark.

The final O1 benchmark applies 96 ordinary starts to every validation release in both banks. The first six releases per setting also receive 192 starts. `scripts/holdings_search.py` and its separate freeze record define this extension. All earlier feasible endpoints remain available, so the extension can only widen existing intervals or resolve empty sets. It leaves the trained models, calibration, response allowance, and full-panel comparisons unchanged. This numerical change followed inspection of validation diagnostics and must not be described as part of the original prespecified validation procedure. Both the original and expanded-search results remain available.

The expanded O1 benchmark remains provisional. The 96-to-192-start comparisons find four material movements in bank 0 and three in bank 1 among 144 checks per bank. The largest is 0.251 response percentage point. Most searches are stable, but these exceptions preclude a claim of dependable endpoint recovery for O1. The linked and unlinked full-panel comparisons have no material movements in their separate 12-to-32-start checks; this still does not prove global optimality.

## Selected calibration follow-up

The three endpoint-audit locations with the lowest observed parameter retention each receive 512 additional independent markets, using a separate seed namespace. This follow-up is diagnostic and selected; it does not refit or recalibrate the models. The report distinguishes the original 128-market estimates from the independent repeats. Exact one-sided binomial comparisons use 96.5% as the observation-test target, with a Bonferroni adjustment across the three fresh repeats. Response-approximation checks use their own simulated means and 99% reference intervals.

## Methods context

Masserano et al. (2023) develop simulator-based confidence regions using conditional test calibration and Neyman inversion. The present statistic remains a forward-summary discrepancy; it is not WALDO. Cabezas et al. (2025, preprint version 2) distinguish finite-sample local calibration from asymptotic conditional coverage. The present three-component rank correction is not an implementation of their TRUST or TRUST++ estimators. These precedents inform calibration and the limits of its claims; the proposed paper's contribution concerns the information supplied by different observations.

1. Masserano, L., Dorigo, T., Izbicki, R., Kuusela, M., and Lee, A. 2023. Simulator-Based Inference with WALDO. Proceedings of Machine Learning Research 206:2960–2974. https://proceedings.mlr.press/v206/masserano23a.html
2. Cabezas, L. M. C., Soares, G. P., Ramos, T. R., Stern, R. B., and Izbicki, R. 2025. Conformal Calibration of Statistical Confidence Sets. arXiv:2411.19368v2. https://arxiv.org/html/2411.19368v2
