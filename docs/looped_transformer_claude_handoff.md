# CREDIT recurrent refinement: handoff for Claude

## User intent and agreed recommendation

The user wants a better, more expressive weather model. The two motivating problems are:

1. CREDIT is only modestly outperforming IFS and remains in a similar regime of predictability.
2. Autoregressive forecasts become blurred and lose small-scale structure.

The user endorsed the recommendation below. This document records a research and implementation proposal; no model or trainer changes have been made for it, and no skill improvements have been demonstrated.

**Start with input-conditioned bottleneck recurrence using two and four passes, alongside a separate experiment with short differentiable rollout windows.** Keep those changes separate initially so their effects are identifiable. If recurrence helps, extend refinement to an intermediate spatial resolution. Defer adaptive halting until additional fixed passes demonstrably improve forecast skill. Pursue conditional probabilistic refinement if remaining blur reflects uncertainty rather than avoidable model error.

The inspiration is recurrent-depth research, not a verified description of GPT-6 internals. The linked article explicitly describes GPT-6 looping as unconfirmed reporting.

## Findings from this checkout

Relevant code:

- `credit/models/wxformer/wxformer_next.py`: `NextGenWXFormer`, `Transformer`, and `SpectralGNNBottleneck`.
- `credit/trainers/trainer_gen2.py`: `_gather_for_next_step` and the per-step backward call in the training loop.
- `credit/losses/base.py`, `credit/losses/__init__.py`, and `credit/losses/gen_1/`: existing loss composition and spectral/ensemble facilities to inspect before extending them.

Corrections to the earlier bottleneck-only proposal:

1. **The bottleneck is not the only place where looping is possible.** Each encoder stage applies a resolution-changing `CrossEmbedLayer` followed by a shape-preserving `Transformer`. The transformer can be repeated without repeating downsampling. Do not loop a shape-changing stage unchanged.
2. **Reusing an existing block does not reduce the current model's parameter count.** It adds computation with approximately unchanged parameters, plus any conditioning parameters. Parameter savings are relative to adding equivalent depth with independent weights. Activation storage and backward computation still grow with unrolled depth unless checkpointing or other techniques are used.
3. **Repeating only `SpectralGNNBottleneck` is restrictive.** It aggregates to learned virtual nodes and scatters corrections through a fixed learned spatial matrix. Its additive corrections remain within that scatter matrix's spatial span. Pair global mixing with spatial attention/local processing. Despite the name, this module is not a spherical-harmonic or FFT operator.
4. **Gen2 currently detaches predictions between physical forecast steps.** `_gather_for_next_step` detaches `y_processed`; the training loop backpropagates each step separately. Training sees model-generated inputs, but later losses cannot differentiate through earlier forecast steps. This is a deliberate memory strategy, not an established bug or a proven cause of blur.

Verify these details against the working tree before implementation. The inspected NextGen model should not be assumed to describe every CREDIT model or every production configuration.

## Experiment A: input-conditioned recurrent refinement

Encode once and repeatedly refine a latent representation before decoding:

```text
c = encoder(input_history)
z = c
for k in range(K):
    delta = shared_refiner(norm(z), c, iteration_embedding(k))
    z = z + residual_gate(k) * delta
prediction = latest_prognostic_state + decoder(z, encoder_skips)
```

This is architectural pseudocode. Adapt it to the model's existing residual prediction path without adding the persistence residual twice. Also avoid accidentally doubling residual connections already internal to reused modules.

- The shared refiner should combine spatial attention, global mixing, and channel processing.
- Reinject the original encoded input through a learned projection or conditioning operation on every pass.
- Use small initialized residual gates to control update magnitude; they do not guarantee stability.
- Keep gradients through all internal refinement passes for the initial two-/four-pass experiments.
- All passes target the **same next physical forecast time**. Four refinement passes do not mean four forecast time steps.
- Preserve encoder skips. Their presence and input reinjection provide access to detail, but do not guarantee its retention.
- Establish fixed-depth behavior first. Then test randomized loop counts during training and evaluate each supported inference budget. Do not assume extra inference loops improve skill or safely extrapolate beyond trained depths.
- Compare against independent blocks with comparable computation. Shared depth can improve over a shallow baseline but is not inherently more expressive than equivalent untied depth.

If bottleneck refinement helps, add a shared refinement block at an intermediate decoder resolution after coarse information has reached it. The purpose is to refine fronts and gradients as well as large-scale circulation. This is a separate ablation, since a coarse bottleneck alone may not address small-scale degradation.

## Experiment B: differentiable rollout windows

Independently test two-step, then four-step windows of backpropagation through physical forecast time:

- Accumulate a weighted loss over the window and backpropagate once per window.
- Preserve gradients through the predicted states used as subsequent inputs, including reconstruction/postprocessing and required distributed communication.
- Detach at window boundaries.
- Use activation checkpointing where needed and measure memory and runtime.
- Preserve existing gradient accumulation, mixed precision, and distributed synchronization semantics.

**Do not implement this by merely removing `.detach()`.** The current per-step backward structure releases graphs, and distributed gather/postprocessing paths must support the intended gradients. Start with the simplest supported execution mode and verify later-step gradient flow into earlier predictions before broadening distributed coverage.

The hypothesis is that later losses can teach earlier steps to preserve dynamically useful information. This may improve rollout skill, but it needs an independent test.

## Diagnose blur before choosing its remedy

Squared-error training targets the conditional mean. When plausible futures place a front at different positions, their mean is broad. More computation cannot recover information missing from the conditioning state.

Separate these possibilities:

| Observation | Candidate response |
| --- | --- |
| Small-scale amplitude decays while locations remain reasonably accurate | Refinement, scale-aware supervision, differentiable rollout training |
| Features remain sharp but are misplaced | Better dynamics and conditioning; sharpening alone is insufficient |
| Several distinct futures are plausible | Conditional probabilistic prediction and coherent ensemble trajectories |

For deterministic experiments, test a modest, scale-normalized gradient or band-limited error term alongside state loss. Use geometry-appropriate operators and variable/scale normalization. Inspect the existing loss plumbing rather than assuming a configuration flag activates the desired objective.

Do not use power-spectrum agreement alone as evidence of improvement: a forecast can have the right energy at the wrong locations. Evaluate spectral coherence or phase-sensitive errors as well. Weighting every small-scale discrepancy aggressively can also punish unavoidable displacement uncertainty.

PDE-Refiner motivates iterative correction of neglected frequency components, but its PDE benchmark results are not evidence of gains on global weather.

For uncertainty-driven blur, the larger research direction is a conditional diffusion/refinement model producing physically coupled next-state samples. Each ensemble member must carry its own sampled trajectory forward. Averaging members before the next forecast step reintroduces smoothing. Evaluate member realism, ensemble calibration, and ensemble-mean accuracy separately. A probabilistic ensemble mean may appropriately remain smooth.

## Experiment order and controls

| Experiment | Main question |
| --- | --- |
| Current model and training | Reference skill, spectra, drift, cost |
| Baseline + differentiable rollout windows | Does temporal credit assignment help? |
| Baseline + conditioned bottleneck recurrence, two/four passes | Does additional shared computation help? |
| Independent blocks with comparable computation | Is recurrence competitive with ordinary depth? |
| Best recurrent model + intermediate-resolution refinement | Does finer-scale refinement preserve useful detail? |
| Best architecture + scale-aware objective | Is deterministic supervision limiting detail? |
| Conditional probabilistic refinement | Can coherent uncertainty modeling improve useful forecast distributions? |

Combine successful changes only after their individual contributions are understood. Compare both training GPU-hours and inference cost; equal optimizer steps alone are not a fair compute comparison. Record parameter count, peak memory, latency, data exposure, and loop count. A warm-start prototype can establish feasibility, but failure to retrofit an existing checkpoint does not conclusively reject the architecture.

## Evaluation and decision criteria

Define success as additional lead time at a fixed useful-skill threshold, stratified by variable, scale, and event. Recurrent computation could reduce model error; it cannot be presumed to remove atmospheric predictability limits.

Evaluate:

- RMSE and ACC by variable, level, and forecast lead.
- Spectral amplitude and coherence, plus front/cyclone position and intensity.
- Long-rollout drift and relevant physical budgets.
- For ensembles: CRPS, reliability, spread–skill behavior, and event probabilities, alongside spatial/temporal member coherence.
- Paired uncertainty estimates across initialization dates, accounting for temporal dependence, on held-out seasons. Use multiple training seeds for decisive comparisons when feasible.

Match IFS forecast products, initialization times, available inputs, verification grids, and valid times. Compare ensemble performance against ENS as appropriate. Treat initialization/data advantages separately from architecture gains.

Continue with recurrence only if additional passes improve forecast skill sufficiently to justify their measured cost. Increased sharpness without improved location-sensitive or probabilistic skill is not a success. Choose quantitative acceptance thresholds before examining final held-out results.

If additional depth saturates, investigate input history, vertical resolution, moist processes, boundary conditions, and initial-state uncertainty. Repeated computation cannot substitute indefinitely for missing information.

## Why adaptive halting comes later

First establish a useful skill-versus-loop-count curve. Adaptive halting adds routing/training complexity, and small updates do not necessarily mean an accurate forecast. Per-token halting may not save wall-clock time with dense kernels or synchronized distributed execution. A global per-forecast budget is a simpler later experiment once fixed loops have demonstrated value.

## References

- [Raschka article: GPT-6 Astra, Looped Transformers, and Hidden Reasoning](https://magazine.sebastianraschka.com/p/gpt-6-astra-looped-transformers-and)
- [Geiping et al.: Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171)
- [Universal Transformers](https://arxiv.org/abs/1807.03819)
- [PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers](https://arxiv.org/abs/2308.05732)
- [GenCast: Diffusion-based ensemble forecasting for medium-range weather](https://arxiv.org/abs/2312.15796)
