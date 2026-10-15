# Bitcoin synthetic evidence pilot expansion

Version 0.3.0. Specification fixed before the independent validation run.

## Scientific question

Can linked evidence on beliefs, available cash, position limits, and risk preferences constrain a Bitcoin price response when several economic mechanisms operate together and participation is uncertain? This is a synthetic identification experiment, not an empirical Bitcoin price forecast. The model contains no narrative diffusion or narrative agent model.

The population, market clearing, intervention, response target, and observation modules remain those of v0.2. All three economic drifts can now vary at once. Five population parameters and the log participation odds ratio are also uncertain. The resulting continuous domain has nine dimensions. Bounds and validation cases appear in `results/expanded_pilot/specification.json`.

## Simulation banks and frozen choices

Two separately generated training and calibration banks use disjoint population, shock, and observation seeds. Each bank contains 1,024 training settings with eight markets per setting; 128 additional emulator-check settings with 12 markets each; 1,999 independent calibration settings with one market each; and 199 response-error settings with 40 markets each. The emulator check is diagnostic and does not choose the degree or tune the estimator.

For training and checks, coordinates are Latin hypercube draws. Calibration and response-error coordinates are independent uniform draws. Independent mechanism labels allocate probability 3/4 to the full mixed domain and 1/12 to each pure-mechanism subspace. Participation odds are log-uniform between 0.25 and 6. This allocation defines a simulation design, not an empirical prior. Values below one favor less optimistic respondents. The mean participation rate remains fixed at one half; selection on unobserved dimensions other than optimism is outside this pilot.

The forward summary emulator is a fixed cubic polynomial in normalized parameters, with ridge penalty 0.1 and an unpenalized intercept. The response emulator excludes participation because participation cannot change the population response. Expected public signals are fixed at zero. Training residuals supply a pooled within-setting covariance, shrunk 10% toward its diagonal. The four panel summaries are unchanged from v0.2, with identical marginal summaries in O6 and O7. No claim of an optimal use of the raw data is made.

## Calibration and its limits

For each panel, the discrepancy is the squared Mahalanobis distance between the released summary and the emulated mean. Its cutoff is order statistic ceil((m+1) times 0.965) of m=1,999 independently calibrated discrepancies. This supplies a finite-sample marginal rank calibration over the declared parameter mixture, conditional on the trained score. It does not establish 96.5% acceptance at every fixed parameter.

An independent response-error envelope accounts for approximation and finite simulator replication. At each of 199 settings, compute the absolute difference between the response emulator and the 40-market mean, then add that mean's two-sided 99.5% Student-t half-width. The envelope is order statistic ceil(200 times 0.99), the second largest of these 199 values. Its interpretation relies on the approximate Student-t coverage of the simulated mean. Combining observation rejection 3.5%, emulator rank error 1%, and reference uncertainty 0.5% yields an approximate 95% design-marginal target **before numerical search error**. This is weaker than pointwise coverage across a continuous domain and is reported explicitly.

For a release, optimize the smallest and largest emulated population response among parameters passing the discrepancy cutoff, then expand both endpoints by the response-error envelope. Search is continuous, using analytic derivatives. Every start first minimizes discrepancy with L-BFGS-B, then searches both response endpoints with constrained SLSQP. Starts alternate proximity to the release with parameter-space separation among plausible global seeds. Each later search budget retains every earlier feasible endpoint. Failed optimizer status is counted; a feasible candidate may remain useful even when the optimizer has not certified convergence. Global optimality is not claimed.

## Validation and search checks

Twelve settings include three new pure cases, six mixed cases, and three boundary stress cases. One mixed setting repeats the previously omitted-mechanism challenge with fresh markets for comparability; the other settings are new. Each setting has 40 independent evaluation markets and a separate 640-market response reference. Models are frozen before generating these validation banks. The evaluator knows truth solely to score results. The isolated-release fitter reads only a trained model and released observations.

Both fitted banks evaluate the same 480 markets so their differences are paired, not 960 independent replications. Main results use eight starts. Two-start results are retained for all releases. The first five replicates at every setting and panel also receive 24 starts. An endpoint movement above 0.005 response percentage points is flagged. This tolerance is a search diagnostic, not a statistical error bound. Confidence intervals for proportions are Wilson intervals; oracle sensitivity uses a separate 99% reference interval. Pooled scenario summaries describe this fixed collection, not a random sample of all parameters.

For O6 only, the first ten replicates per setting also receive a separately labeled known-participation fit. Only this benchmark fixes the true participation strength. Its cutoff remains the same marginal cutoff, so the benchmark is a sensitivity analysis without a separate conditional coverage claim.

An additional simulator audit checks the lower and upper endpoints from bank 0 O6 for the first release in each validation setting, using fresh audit seeds. It measures discrepancy rejection and response approximation at the selected endpoints. Because these endpoints are selected, response-envelope calibration alone does not guarantee their accuracy. The audit is diagnostic and does not feed back into the frozen reported fits.

## Decision rule for full confirmation

Proceed only if the expanded model materially resolves the omitted-mechanism failures, the linked-versus-unlinked comparison survives both fitted banks, and search changes are small. Even if these conditions hold, a pointwise coverage claim requires stronger conditional calibration or a justified restriction of the inferential claim. Fixed-case coverage checks and selected endpoint audits must be reported, including failures. A larger repeat count cannot correct an inadequately calibrated procedure. The earlier reserved confirmation seeds remain unused.

## Methods context

The distinction between design-marginal and parameter-conditional coverage is central. The current implementation is a forward-emulator projection with marginal rank calibration; it is not the WALDO estimator. A conditional critical-value emulator with independent coverage diagnostics is a candidate next improvement if pointwise confidence is retained as the paper's objective. Relevant primary sources are Masserano et al. (2023), *Simulator-Based Inference with WALDO*, Proceedings of AISTATS 206:2960–2974, https://proceedings.mlr.press/v206/masserano23a.html; and Shafer and Vovk (2008), *A Tutorial on Conformal Prediction*, JMLR 9:371–421, https://jmlr.org/papers/v9/shafer08a.html.
