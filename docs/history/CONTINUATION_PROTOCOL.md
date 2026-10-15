# Bitcoin synthetic evidence continuation

Version 0.5.0. Single bounded repair from the saved version 0.4.0 package.

## Question and scope

Can a conservative, independently simulated observation cutoff and a response-directed endpoint search remove the specific calibration and numerical failures found in version 0.4 without eliminating the practical benefit of linked observations? The economic model, observation modules, nine-dimensional parameter domain, response approximation, and response-error allowances remain unchanged. The target is the synthetic population mean month-19 price response to a transfer of 5% of aggregate month-18 cash. This exercise does not estimate a real Bitcoin response.

All earlier results now serve as development information. The original confirmation seeds remain unused. There is one frozen procedure and one fresh validation pass; validation failures will not trigger tuning within this release. Machine-readable counts, seeds, locations, budgets, and decision gates are in `results/continuation_v05/specification.json`.

## Calibration repair

For each inherited fitted bank, select the six historical audited endpoints with the lowest observed parameter retention, then six more historical endpoints by farthest-point allocation in normalized parameter space. The resulting 12 locations include the previously confirmed failures. Simulate 512 new independent markets at each location using a fresh seed namespace. The two banks use different calibration markets. Each market supplies all four observation panels.

Let R be the original squared-discrepancy score divided by its original parameter-dependent cutoff. At each anchor and for each panel, use the 507th ordered R value among 512 simulations. This is the smallest rank k for which Pr[Binomial(512, 0.965) <= k-1] is at least 1 - 0.05/96. There are 96 bank-panel-anchor combinations. A union bound therefore gives at least 95% simultaneous confidence that these 96 upper tolerance limits each contain at least 96.5% of their respective score-ratio populations, conditional on the inherited models and selected locations. Continuity gives the usual exact order-statistic statement; ties make the inequality conservative.

For each bank-panel model, multiply its original cutoff by the maximum of 1 and its 12 tolerance limits. The multiplier is constant across parameters, preserving the smooth cutoff and its derivatives. At every selected anchor it is at least as protective as that anchor's tolerance limit. At all parameters the accepted set expands or stays unchanged. Consequently the original component-marginal observation-acceptance protection is not weakened. This is a mathematical set-inclusion statement before numerical inversion, not a claim that an approximate search always finds the expanded set.

The finite-anchor confidence statement does not extend automatically to untested parameter values. It also does not certify response coverage: projection search and the inherited response approximation introduce separate uncertainties. The repair may be conservative and may reduce precision. Report its interval-width cost and linkage comparison explicitly.

## Search repair

The original routine mostly selected starts near the observations and first moved every start to a discrepancy minimum. That can lose useful paths to remote response extrema. The new routine uses 8,192 scrambled Sobol locations, all 512 corners, and the inherited training locations. It alternates starts directed toward fit, low responses, high responses, and broad parameter-space coverage. Direct constrained endpoint optimization precedes selective discrepancy minimization. Every accepted endpoint must satisfy the original feasibility tolerance and parameter bounds; optimizer termination codes remain diagnostic.

Main budgets are 64 starts for holdings alone and 24 for each other panel. Increasing to 128 or 48 starts retains earlier endpoints. The first two released markets at every validation setting receive both budgets. The first holdings-only market at every setting also receives an independent differential-evolution check with 96 population members, 160 generations, fixed random seeds, and local constrained polishing. This check can only expand the main interval. Neither procedure certifies global optimality.

Before freezing, the new search is checked against the seven historical fits that moved by more than 0.005 percentage point between the old 96- and 192-start searches, plus a constructed disconnected feasible set with known endpoints. Those are development checks, not fresh validation results.

## Fresh validation

Freeze and hash the inference source, design, and eight models before generating validation. Use 36 settings: the 24 historical settings, the three selected historical follow-up endpoints, and nine new locations allocated equally to bulk, faces, and corners. At every setting simulate 512 independent validation markets and a separate 512-market reference mean. Both banks evaluate the same validation markets; they are paired model comparisons rather than additional independent markets.

Parameter retention uses all 512 validation markets at each setting. Interval inversion uses the first 16 markets at each setting, giving 576 independent market realizations per panel and bank. Reproduce and release their raw observation panels under opaque filenames. Fitters receive only these protected releases and their portable models. The larger evaluation records and generating parameters remain evaluator-only. Participation remains unknown to every primary fit.

After fitting, audit the main lower and upper endpoints for panels O1, O6, and O7 at six prespecified settings in both banks. Each of the 72 selected locations receives 128 fresh simulations. Check the response approximation against the independently estimated mean and its 99% Student-t interval; also report observation-score retention at these newly selected endpoints. These endpoint checks are targeted diagnostics and do not estimate the frequency of failures across the domain.

## Preset decision gates

1. No simulation failures, accounting residuals below 1e-8, and exact reproduction of all 2,304 released observation summaries.
2. Observed parameter retention of at least 96.5% in each of the 288 bank-panel-case cells. Additionally, the one-sided exact 95% lower confidence bound must reach 96.5% at each of the three focal O6 locations in both banks. These are demanding finite-case gates, not continuous-domain guarantees.
3. For each bank-panel combination, pooled response coverage must have a one-sided exact 95% lower bound of at least 95%; each 16-market cell must cover at least 15 reference means. Also report sensitivity to uncertainty in those reference means.
4. No empty-set reversal and no endpoint movement above 0.005 percentage point in any larger-budget or differential-evolution comparison.
5. All 72 selected response-approximation allowances must contain the independent endpoint means and their 99% reference intervals.
6. Report paired linked-versus-unlinked width differences, Monte Carlo errors, coverage, and empty sets. Positive linkage gains are not required to pass.

Any failed gate keeps the large confirmation run on hold. Even passing these gates supports only the stated finite experiment. A broader confirmation protocol requires a separate decision about the paper's inferential claims.

## Statistical and computational sources

The tolerance-limit construction follows the distribution-free order-statistic principle associated with Wilks (1941), Determination of Sample Sizes for Setting Tolerance Limits, Annals of Mathematical Statistics 12, 91-96. The binomial identity and union-bound extension used here are stated explicitly above. NIST distinguishes population content from the confidence attached to a tolerance limit.[^1]

The independent global check uses SciPy differential evolution with constraints and vectorized evaluation.[^2] Search agreement is empirical evidence about numerical stability and is not a proof of global optimality.

## Reporting clarification after the frozen computation

The frozen pooled response gate refers to an exact binomial bound. Because coverage probabilities can differ across the 36 fixed settings, that pooled calculation is a binomial reference rather than an exact homogeneous-binomial confidence statement. Its values and the prespecified gate remain in the results. The report additionally uses a conservative Chernoff lower bound valid for independent Bernoulli indicators with unequal probabilities; this additional check also passes. It does not change any inference model, cutoff, interval, validation seed, search budget, or gate threshold.

For independent indicators with probabilities p_i and average p_bar, concavity of log gives E[exp(t sum X_i)] <= (1 - p_bar + p_bar exp(t))^n. Markov's inequality and optimization over t > 0 give Pr[mean X >= q] <= exp(-n D(q || p_bar)) for q > p_bar, where D is Bernoulli relative entropy. The one-sided 95% lower bound inverts n D(observed fraction || p_lower) = log(20). For all successes it equals 0.05^(1/n). The calculation conditions on the fitted models and independent reference means; the separate definite/possible coverage calculation assesses reference-mean uncertainty.

Paired linkage uncertainty uses a stratified delta-method calculation, conditional on the fixed locations and fitted models. For each market, set U = I_nonempty (width_difference - estimated_mean_difference); empty pairs contribute zero. The estimated variance is the sum over settings of n_setting times the within-setting sample variance of U, divided by the squared total number of nonempty pairs. The original pooled pair standard error is retained as a descriptive comparison. These calculations do not estimate uncertainty from selecting parameter locations or refitting the models.

[^1]: https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/tolelimi.htm
[^2]: https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.optimize.differential_evolution.html
