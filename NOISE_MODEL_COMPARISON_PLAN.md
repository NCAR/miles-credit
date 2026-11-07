# Noise-Model Comparisons for SDL CAMulator: Broad Questions and Hypotheses

**Status:** draft plan, 2026-11-06. For discussion; nothing here is adopted.

**What this is.** A small set of broad comparisons. Each runs several noise models through the same
experiments and scores them against CESM and against each other. Each is chosen so that *either* outcome
points to a specific, more targeted study. The hypotheses are bets with reasons, not expected results.

**How it relates to the other documents.** This is the comparison layer on top of:
- `SDL_STOCHASTICITY_RESEARCH_PATH.md`: experiments E1–E11, questions Q1–Q13, the three-paper arc. Its
  conventions (§2) apply here: the ensemble mean only for expected values, per-member statistics for variability
  and extremes, the full distribution for calibration.
- `SDL_CHAOS_SOBOL_ROADMAP.md`: RQ1–RQ6, the toy theory (§4) and the residual findings (§10.1).
- The red-noise hand-over note (implementation and local results).

Each question below lists the E, Q and RQ numbers it overlaps.

---

## 1. Starting point

**CAMulator's one-step errors** (held-out 2013–14; roadmap §10.1):
- Free atmosphere and precipitation: memory of about a day or less.
- Surface over land and sea ice: a slow tail, 10–20% of the error variance, timescale about 1.5–4.5 weeks.
  It is not predictable from SST, sea ice or Niño3.4, so it is probably the land state the emulator doesn't carry.
- Strongly state-dependent: precipitation error spread is 4–6× larger in the most active third of cases than
  the quietest; land fluxes 1.6–1.7×; U and T at 850 hPa about 1.3×.
- Anti-correlated with the true 6-h change (about −0.2 to −0.3): the emulator under-predicts change, the
  smoothing typical of MSE training.

**Plain CAMulator from a perfect start** (local, with the MPS fix): one-step errors match the residual analysis,
and global-mean TREFHT drifts only −0.26 K over 30 days from 1981-01-01.

**Land noise on plain CAMulator** (local, 3 members, 30 days): red land noise gives 2.6× the land TREFHT spread
of white noise of the same per-step size at day 10 (0.92 vs 0.36 K). White spread stays flat for 10 days.

**White SDL** (Derecho; epochs 1–10 done, 11–20 queued): rough training-log estimates of one-step spread/error
rose to about 0.8–1.0 (T 0.82, U 0.91, PRECT 0.93, TREFHT 0.95). To be checked on the epoch-20 checkpoint.

**One fact shapes everything below: the hindcasts start from the exact CESM state.** There is no initial-condition
error, so all forecast error is model error (emulator vs CESM), amplified by chaos. The noise models are
competing to represent model error, which is the point.

---

## 2. The noise models

Three choices define a noise model:
- **Who sets it:** learned in training, or sized from the residual analysis.
- **Form:** additive, or multiplicative (scaled with the size of the step's change).
- **Memory:** white (fresh every 6 h) or red (persists for days to weeks).

A fourth is **where** it enters: inside the network, on the model's output before the conservation fixers
(so it feeds back into the next step), or only on outputs that don't feed back.

| Name | What it does | Set by | Form | Memory | Status |
|---|---|---|---|---|---|
| **Plain CAMulator** | No noise; one member | — | — | — | ready |
| **Start-perturbed** | Plain CAMulator from slightly perturbed starts, sized like one step of error | residuals | additive, once | — | built (`start_perturbation` preblock) |
| **White SDL** | Noise inside the decoder, fresh each step | training (CRPS) | additive in feature space* | white | training, epoch 20 |
| **Red SDL** | Same weights; the SDL noise gets memory | training + chosen τ | additive* | red | built (config switch) |
| **Land noise** | Slowly varying nudges to near-surface T and humidity over land and sea ice, after the model, before the fixers | residuals | additive | red, 11 d (T) / 14 d (q) | built |
| **White land noise** | Land noise without memory, amplitude raised to match red's week-1–2 spread | residuals | additive | white | built (scaled copy of the sigma maps) |
| **Tendency noise** (SPPT-style) | Scales each step's change by (1 + r), with r a smooth, multi-scale red random field; after the model, before the fixers | NWP defaults or residual fit | multiplicative | red, multi-scale | built (`tendency_noise` postblock) |
| **Multiplicative SDL** | SDL noise that scales the decoder features instead of adding to them | training | multiplicative in feature space | white or red | needs code and a training run (later) |
| **Output dressing** | State-dependent noise on outputs that don't feed back (precipitation, fluxes) | residuals | multiplicative | white | built offline (`score_hindcasts.py --dress`); amplitude to fit |
| **Warm-started land noise** | Land noise whose memory starts from the actual errors of the ~5 days before the start | residuals + truth | additive | red | to build; perfect-model only |

\* Additive in the decoder's feature space. The decoder is nonlinear, so the effect in physical space may still
depend on the state. Measuring that is question 1.1.

Combinations used below: White SDL + land noise, White SDL + tendency noise, Plain + land noise + tendency noise.

### 2.1 Design notes for the pieces to build

**Start perturbations** (decided 2026-11-06: state differences). `start_perturbation` adds
amplitude × (s1 − s2) / √2 to every prognostic, with s1 and s2 two CESM states at the start's hour and within
15 days of its calendar day, from the training years 1980–2012.
- One amplitude for all variables keeps the kick balanced. On 1981 states, one-step RMSE / RMS((s1 − s2) / √2)
  is about 0.21 for T, 0.12 for U, 0.26 for Qtot, 0.28 for TREFHT and 0.06 for PS.
- The default 0.15 gives T and winds a kick near their one-step error, PS about 2.7× and TREFHT and humidity
  about half. Refit on Casper with the real donor years.
- Rejected: per-variable random fields (unbalanced, so part of the kick decays) and bred vectors (the most
  work; gen1's `bred_vector.py` is not wired into gen2 rollout).

**Tendency noise** (decided 2026-11-06: T, Qtot, U and V below ~100 hPa, CAM levels 10–31, for the first wave).
- Update: x(t+1) = x(t) + (1 + r) · (prediction − x(t)) for prognostic T, Qtot, U and V. The pattern r is the
  sum of three independent patterns, clipped to [−1, 1].
- Starting values from ECMWF's three-scale SPPT (☐ verify):
  - standard deviation 0.52 / 0.18 / 0.06;
  - timescale 6 h / 3 d / 30 d;
  - length scale 500 / 1000 / 2000 km.
- Alternative amplitude: fit it from the residual data by regressing residual spread on the size of the
  *predicted* change. The residual archive stores the error and the true change; the predicted change is their
  sum.
- Two caveats:
  - The emulator's 6-h change includes the resolved dynamics (advection), which SPPT in a GCM leaves alone.
  - Perturbing winds may unbalance the flow.
  
  A thermodynamic-only variant (T and Qtot only) tests the second.

**Output dressing** needs no new runs: it can be applied to any arm's saved output in the scorer.

**Warm start.** Before each start date, run one-step forecasts over the previous ~5 days. Their errors give the
land-noise state at t = 0.

---

## 3. What we compare against

- **CESM truth for 2013–14** (held out from training): hindcasts and one-step portraits.
- **The one-step residuals:** the target for "is this the right noise?".
- **CESM 1980–2014 statistics** for climate comparisons. These are training years, which matters less for
  statistics than for trajectories.
- **Each other, paired:** the same starts and the same seeds (common random numbers), so a difference between
  two noise models isn't sampling noise.
- ☐ **A CAM/CESM initial-condition ensemble with the same SST** would give the internal variability that spread
  should saturate at (roadmap §4.3). Not located yet.

---

## 4. Three run suites

Every question draws on three suites; none needs its own runs.

| Suite | What | Steps per noise model | Answers |
|---|---|---|---|
| **One-step portraits** | ~500 CESM states across seasons × 50 noise draws, one 6-h step, plus noise off | ~25k | 1.1, 1.3 |
| **Hindcasts** | 99 weekly starts (2013-01-01 to 2014-11-18) × 10 members × 42 days | ~166k (plain: 17k) | 1.2, 2.1–2.8 |
| **Long runs** | 3 members × 10 years with CESM forcing (or 1 × 35 years) | ~44–51k | 3.1–3.3; baseline for 4.1 |

**Hindcast arms** (`hindcast_configs.py`, named after the noise models):
- **First wave (the default):** `plain`, `plain_kick` (start-perturbed), `sdl`, `sdl_red`, `sdl_land`,
  `plain_land`, `plain_sppt` (tendency noise).
- **Second wave, after a first look:** `sdl_land_white` and `plain_land_white` (spread-matched, amplitude ×2.4
  from the local runs), `sdl_sppt`, `plain_sppt_thermo`, `plain_land_sppt`, `sdl_kick`; warm-started land noise
  is not built yet.
- Output dressing is applied offline (`score_hindcasts.py --dress ARM --dress-sigma s`).

**Hindcast output: unchanged.** Seven near-surface 2-D fields at 00Z; PRECT is the 6-h accumulation ending at 00Z.
- The rollout writer cannot keep only some levels of a 3-D field when the data config has no level list, as in
  CAMulator's configs. It would write all 32 levels, about 10× the storage.
- MJO phase and regimes are only needed at the start date, for conditioning (2.7). They come from the CESM
  truth, not from the forecasts.
- Daily precipitation totals and extremes (3.3) come from the long runs instead.

**Scorer additions** (`score_hindcasts.py`):
- spread/skill in thirds of each region by the ensemble-mean forecast (quiet / middle / active, 2.3);
- fair Brier score for PRECT above 2.5 and 6.25 mm per 6 h (2.3);
- weekly error and spread per start date, for the spread–error correlation across starts (2.7);
- output dressing.

**Cost.** About 1.0M steps for the 7 first-wave hindcast arms, 0.7M for the second wave, 0.3M for the long runs
(6 models) and 0.2M for the portraits. At an assumed 0.25 s per step on one A100 (☐ measure on Casper first),
that is about 70, 45, 20 and 15 GPU-hours. Members can be batched.

**Statistics.**
- Paired differences between noise models.
- Bootstrap over start dates in blocks of about 4 weeks, since the 42-day forecasts from weekly starts overlap.
- Skill-score differences with 90% intervals.
- Expect calibration results to be clearer than skill results.

---

## 5. Question / hypothesis matrix

### 5.1 Overview

| ID | Broad question | Main comparison | Headline hypothesis | Suite |
|---|---|---|---|---|
| **1. Is it the right noise?** | | | | |
| 1.1 | Does the learned noise look like CAMulator's actual errors? | White SDL vs one-step residuals | Right average size, but too weak aloft, too little state dependence, and spread maps only partly match error maps | portraits |
| 1.2 | Which of plain CAMulator's errors persist through a rollout? | Plain vs CESM | The persistent part sits at the land and sea-ice surface; the free atmosphere has no systematic drift | hindcasts |
| 1.3 | How physical is each noise? | All, using the fixers and wind filter as a meter | Tendency noise forces the largest budget corrections; land noise the smallest | portraits, hindcasts |
| **2. Forecast value, days to weeks** | | | | |
| 2.1 | Is continual noise better than a one-time kick? | White SDL vs start-perturbed vs both | SDL better calibrated to ~2 weeks; both converge by weeks 3–4 | hindcasts |
| 2.2 | Does memory help, and only where errors have memory? | White SDL vs red SDL vs SDL + land noise vs SDL + white land noise | Targeted memory beats blanket memory | hindcasts |
| 2.3 | Does multiplicative noise give state-dependent reliability? | Additive vs tendency noise vs output dressing | Additive noise is too wide when quiet and too narrow when active; multiplicative evens this out, most for precipitation | hindcasts |
| 2.4 | Is learned noise better than noise sized from the residuals? | White SDL vs plain + land + tendency noise vs both | Learned wins days 1–5; residual-sized competitive from week 2; together best | hindcasts |
| 2.5 | Does the week-1 ranking predict the week-4 ranking? | All | No: free-atmosphere scores converge, land-surface ranking set by memory | hindcasts |
| 2.6 | Is S2S spread limited by the noise or by the emulator's own variability? | Saturated spread vs long-run variability vs CESM | Limited by variability: every noise model is under-dispersed at weeks 4–6 unless it also raises variability | hindcasts, long runs |
| 2.7 | Do the noise models know when to be confident? | All, across start dates | The flow decides which starts are confident; noise models differ mainly in average spread | hindcasts |
| 2.8 | Can the noise carry information, not just spread? | Land noise vs warm-started land noise | Small but real land skill gain in weeks 1–3 | hindcasts |
| **3. The emulated climate** | | | | |
| 3.1 | Does noise shift the mean climate, toward or away from CESM? | All long runs vs plain vs CESM | White SDL shifts little; red and multiplicative noise shift more | long runs |
| 3.2 | Which noise fixes which part of the variability spectrum? | Per-member spectra, all vs CESM | Plain is too smooth; white noise fixes fast variance, red the slow, multiplicative the precipitation; none fixes all | long runs |
| 3.3 | Extremes, persistent events and regimes | All vs CESM | Tails too weak in plain; red land noise lengthens warm/dry spells, maybe too much | long runs |
| **4. Forced response (later)** | | | | |
| 4.1 | Does the noise model change the response to a forcing? | Plain vs best noise model, +2 K SST | Only if 3.2 shows large variability changes | long runs (paired) |

### 5.2 Theme 1: Is it the right noise?

#### 1.1 Does the learned noise look like CAMulator's actual errors?
- **Compare:** White SDL's one-step spread (noise draws around the noise-off step) with the one-step residuals.
  Red SDL is identical at one step.
- **Hypothesis:** about the right average size where CRPS weighted heavily, but three mismatches:
  - too weak aloft: the training-log ratios and the SDL paper both point to under-dispersion in upper-level
    winds and temperature;
  - weaker state dependence than the errors: an active/quiet precipitation spread ratio well below 4–6, because the
    noise amplitude per feature channel is the same everywhere and state dependence enters only through the
    decoder's nonlinearity;
  - spread maps only partly correlated with error maps, because the decoder's sensitivity, not the error, decides
    where the noise lands.
- **Measure:**
  - spread/error maps by variable and level;
  - active/quiet spread ratio compared with the residuals';
  - pattern correlation of spread and error maps;
  - correlation length;
  - T–humidity and wind balance of the noise.
- **If it matches:** SDL is a credible learned stochastic parameterization, Paper 1's main claim.
- **If too additive:** the case for multiplicative noise (2.3, Multiplicative SDL).
- **If wrong place:** a spatially varying amplitude, learned or taken from the residuals.
- **Links:** E1–E2, Q1–Q2.

#### 1.2 Which of plain CAMulator's errors persist through a rollout?
- **Compare:** Plain CAMulator hindcasts with CESM. No noise needed; this can run first.
- **Hypothesis:** the persistent, systematic part of rollout error sits at the land and sea-ice surface, where
  the one-step slow tail is, and grows over the first 2–3 weeks. Free-atmosphere error grows chaotically and
  saturates without a systematic drift.
- **Measure:**
  - bias by lead, region and variable;
  - error autocorrelation across leads;
  - how much of the error is shared by starts in the same season (systematic) vs not.
- **If confirmed:** supports targeted land noise and the warm start (2.8).
- **If the free atmosphere drifts too:** that part needs a deterministic correction, not noise (decision log,
  2026-10-27).
- **Links:** E4, Q5, RQ6.

#### 1.3 How physical is each noise?
- **Compare:** every noise model with plain CAMulator, using CAMulator's own mass, water and energy fixers and its
  wind filter as a meter.
- **Hypothesis:** noise added before the fixers changes the budgets, and the fixers undo part of it.
  - Land noise (near-surface T and humidity) triggers the water fixer, lightly.
  - Tendency noise on winds and temperature triggers the largest mass and energy corrections.
  - SDL noise adds small-scale texture that the wind filter partly removes.
- **Measure:**
  - each fixer's correction and the wind filter's change per step, relative to plain;
  - spread before vs after the fixers;
  - spatial spectra of the perturbations.
- **If corrections are large:** part of the spread is being undone and the injection point should move, for
  example to thermodynamic-only tendency noise or to output dressing.
- **If small:** the noise is consistent with the model's budgets, a claim worth making in Paper 1.
- **Links:** new. Uses per-stage hooks like the one-step diagnostic's.

### 5.3 Theme 2: Forecast value, days to weeks

#### 2.1 Is continual noise better than a one-time kick?
- **Compare:** White SDL vs start-perturbed (the same size of kick, once) vs both; 10 members each.
- **Hypothesis:**
  - Start-perturbed is under-dispersed from day 1 until saturation: model error keeps entering every step,
    and a one-time kick misses it.
  - White SDL is better calibrated through about 10–15 days.
  - By weeks 3–4 both reach the same saturated spread, set by the emulator's own variability (2.6).
  - Adding start perturbations to SDL changes little, because the start is perfect.
- **Measure:** spread/skill and CRPS by lead (daily to week 6), by variable and region.
- **If SDL still wins at weeks 3–6:** model error shapes S2S spread, not just chaos; follow up with the split by
  lead (E6).
- **If they converge by week 2:** the noise's value is in the first two weeks, and S2S work should focus on the
  emulator's variability and slow error (2.2, 2.6).
- **Links:** E5–E6, Q7–Q8.

#### 2.2 Does memory help, and only where the errors have memory?
- **Compare:** White SDL; Red SDL (memory everywhere); White SDL + land noise (memory only at the land surface);
  White SDL + white land noise (spread-matched).
- **Hypothesis: targeted memory beats blanket memory.**
  - Land noise improves weekly-mean land TREFHT and TS calibration in weeks 1–4, where White SDL is
    under-dispersed, and leaves the free atmosphere alone.
  - Red SDL over-disperses the free atmosphere in weeks 1–2: its amplitude was trained for white noise, and
    persistent forcing accumulates. It adds little later, because free-atmosphere errors have no memory.
  - Spread-matched white land noise matches red in week 2 by construction. It falls behind in weeks 3–4 and does
    worse at day 1, since it needs a bigger kick for the same spread.
- **Measure:**
  - weekly-mean CRPS and spread/skill by region (land, ocean, sea ice, tropics, extratropics) and level;
  - how spread grows: the log–log slope separates diffusive from ballistic growth (roadmap §6).
- **If confirmed:** the memory belongs to the missing land state. Next: tune τ and amplitude on long rollouts,
  and test the warm start (2.8).
- **If blanket memory wins:** memory acts through slow atmospheric modes (RQ4). Revisit error memory in rollouts
  (1.2) and consider training with red noise.
- **If neither helps:** land under-dispersion isn't what limits S2S calibration (see 2.6).
- **Links:** E7, Q6, RQ4.

#### 2.3 Does multiplicative noise give state-dependent reliability?
- **Compare:** additive noise (White SDL, land noise) vs tendency noise (on plain CAMulator and with White SDL)
  vs output dressing (multiplicative, no feedback).
- **Hypothesis:**
  - Additive noise is over-dispersed in quiet conditions and under-dispersed in active ones.
  - Multiplicative noise evens this out, with the biggest gain in precipitation.
  - Output dressing fixes the precipitation distribution at short leads, but nothing that depends on feedback,
    such as the circulation's response to heating.
- **Measure:**
  - spread/skill and reliability in terciles of a *forecast* quantity (ensemble-mean precipitation or the size
    of the predicted change), not the truth, so the test is fair;
  - Brier score for heavy precipitation;
  - per-member precipitation distribution vs CESM.
- **If confirmed:** the case for Multiplicative SDL (a training run), or a combined additive–multiplicative
  noise (Sardeshmukh & Penland 2015).
- **If tendency noise ≈ dressing:** precipitation needs noise only on the output, which is cheaper and cannot
  destabilize the model.
- **If additive SDL already matches:** the decoder's nonlinearity makes SDL state-dependent (ties back to 1.1).
- **Links:** roadmap §10.1 item 3; plan v3 arms 2 and 4.

#### 2.4 Is learned noise better than noise sized from the residuals?
- **Compare:** White SDL vs Plain + land noise + tendency noise (no training) vs White SDL + land noise.
- **Hypothesis:**
  - Learned noise wins in days 1–5, where it was directly optimized.
  - Residual-sized noise is competitive from week 2 and in climate statistics, because it carries the memory and
    state dependence that white, one-step training cannot.
  - The combination is best.
- **Measure:** CRPS skill score by lead window, paired on the same starts and seeds.
- **If residual-sized ≈ SDL:** a cheap, interpretable alternative to SDL training for any emulator, which is a
  practical result on its own.
- **If SDL clearly wins:** its latent space captures error structure the residual statistics miss; worth
  explaining in Paper 1.
- **Links:** new.

#### 2.5 Does the week-1 ranking predict the week-4 ranking?
- **Compare:** every noise model.
- **Hypothesis:** no.
  - At week 1 the order follows how well each model matches the one-step error.
  - By weeks 3–6, free-atmosphere scores converge as chaos takes over, and the land-surface order is set by
    memory.
- **Measure:** each model's rank by CRPS skill score per lead window and region, with bootstrap intervals.
- **If rankings hold:** tuning noise at short range is enough.
- **If they flip:** S2S needs objectives beyond one-step CRPS: multi-step training, or tuning τ and amplitude on
  long rollouts. This is the "trained at 6 hours, used at 6 weeks" tension (stochasticity path §1.3) in one plot.
- **Links:** Q7.

#### 2.6 Is S2S spread limited by the noise or by the emulator's own variability?
- **Compare:** each model's spread at saturation (weeks 4–6) with its own long-run variability (3.2) and with
  CESM.
- **Hypothesis:**
  - At saturation, spread equals the emulator's internal variability given the SST.
  - Plain CAMulator is probably too smooth; its errors are anti-correlated with the true change, a sign of MSE
    damping.
  - So every noise model is under-dispersed at weeks 4–6 unless it also raises the emulator's variability.
- **Measure:** spread/skill at weeks 4–6; per-member variance of weekly means in the long runs vs CESM.
- **If confirmed:** S2S calibration is a climate-variability problem; the noise models that fix 3.2 are the ones
  to use at S2S.
- **If spread saturates correctly:** the noise only has to get the approach right.
- **Links:** roadmap §4.3, E9.

#### 2.7 Do the noise models know when to be confident?
- **Compare:** every noise model, across start dates.
- **Hypothesis:**
  - Which starts get narrow or wide ensembles is set mainly by the flow, and is shared across noise models.
  - The noise models differ mostly in average spread.
  - Multiplicative noise may add spread variation of its own, real or spurious.
- **Measure:** correlation of spread and error across starts, per lead window; reliability split by season,
  ENSO phase and MJO activity.
- **If the noise model shapes confidence:** any forecast-of-opportunity result depends on it, so the Sobol work
  must treat the noise model as structural uncertainty (RQ3, RQ5).
- **If not:** the dynamics decide confidence, and the noise choice is about calibration only.
- **Caveat:** 99 starts over two years cover at most one ENSO cycle. Without more held-out data (☐), these
  results are tentative.
- **Links:** Q9, RQ3, RQ5.

#### 2.8 Can the noise carry information, not just spread?
- **Compare:** land noise vs warm-started land noise.
- **Hypothesis:** a small but real skill gain over land in weeks 1–3: a lower ensemble-mean error and CRPS, not
  just more spread. It should be smaller than in L96 (30–100%), because the slow share here is only 10–20%.
- **Measure:** ensemble-mean RMSE and CRPS skill score over land, paired against ordinary land noise.
- **If confirmed:** the noise state stands in for missing land state. Real forecasts would then need to estimate
  it by data assimilation, or the emulator needs land variables; the E3SM emulator with land is the natural
  comparison.
- **If not:** the slow error is too weak or too unpredictable to exploit.
- **Links:** RQ6, L96 notebook 04.

### 5.4 Theme 3: The emulated climate

#### 3.1 Does noise shift the mean climate, toward or away from CESM?
- **Compare:** long runs of every noise model vs plain vs CESM.
- **Hypothesis:**
  - White SDL shifts the mean little.
  - Red and multiplicative noise shift it more. Noise-induced drift grows with the noise's integrated variance,
    and multiplicative noise causes drift even in a linear system.
  - Land noise may make land wetter on average, because saturation and precipitation respond nonlinearly to
    moisture. The direction is speculative; that's what we test.
- **Measure:**
  - long-run mean maps and global and land means of TREFHT, PRECT, column water and surface pressure;
  - jet position;
  - each compared with plain, with significance from the member scatter.
- **If toward CESM:** the noise compensates for something missing (Palmer's hypothesis, tested in an emulator);
  follow up on which process.
- **If away:** drift needs a counter-measure (smaller amplitude, a mean correction) before noise is used for
  climate or forced-response work.
- **Links:** E10, Q11, roadmap §4.5.

#### 3.2 Which noise fixes which part of the variability spectrum?
- **Compare:** per-member variance by timescale (6 h to interannual) and by spatial scale, all models vs CESM.
- **Hypothesis:**
  - Plain CAMulator under-varies, most at short timescales and in extremes (MSE damping).
  - White noise adds variance mostly at fast timescales. Red noise pumps slow modes: the toy model (roadmap §4.2)
    predicts about 27× more variance in a 10-day mode from 20-day red noise than from white. So red land noise
    should restore monthly land variance.
  - Tendency noise restores precipitation variance and skewness.
  - No single model fixes every band.
- **Measure:** per-member frequency spectra; variance of daily, weekly and monthly means; spatial power spectra;
  per-member skewness.
- **If different noises fix different bands:** build a combined, multi-scale noise model, as ECMWF's three-scale
  SPPT does.
- **If one fixes all:** a simpler story and a clear default.
- **Links:** E9–E10, Q10, roadmap §4.2.

#### 3.3 Extremes, persistent events and regimes
- **Compare:** all models vs CESM.
- **Hypothesis:**
  - Plain CAMulator has too-weak tails and too-short persistent events.
  - Multiplicative noise fixes precipitation tails.
  - Red land noise lengthens warm and dry spells, possibly too much.
  - Regime frequencies change most with red noise (Palmer: forcing changes regime probabilities).
- **Measure:**
  - per-member tail quantiles of TREFHT and daily PRECT;
  - warm-spell and dry-spell durations;
  - blocking or regime frequency from U250 or T500 (needs the extra output fields).
- **If events run too long:** the memory is too long or too strong; retune τ.
- **If regimes shift:** a regime-focused study; the L96 work (RQ1) shows what to look for.
- **Links:** E10, Q10, RQ1.

### 5.5 Theme 4: Forced response (later, higher risk)

#### 4.1 Does the noise model change the response to a forcing?
- **Compare:** plain vs the best one or two noise models, paired long runs with and without a uniform +2 K SST,
  several seeds each.
- **Hypothesis:** fluctuation–dissipation reasoning says changing the variability changes the response. If 3.2
  shows large variability changes, expect a detectable change in the response; otherwise not.
- **Gate:** only worth running if 3.1–3.2 show the noise changes the climate. Ties to the SST-response program
  (cluster C4).
- **Links:** E11, Q12.

---

## 6. Decision points: where the answers lead

| Decision | Decided by | If yes | If no |
|---|---|---|---|
| Train a multiplicative SDL? | 1.1, 2.3 | SDL is too additive and tendency noise helps → build it and train | SDL already state-dependent, or dressing suffices → skip |
| Train with red noise, or with multi-step objectives? | 2.2, 2.5 | Rankings flip and memory helps beyond land → train | Targeted inference-time memory is enough → tune τ and amplitude only |
| Pursue the missing land state? | 1.2, 2.8 | Warm start adds skill → data assimilation of the noise state, or land variables in the emulator | Treat land error as noise only |
| Use noise in climate runs and forced-response work? | 3.1–3.3 | Drift small, variability improved → 4.1 and the SST-response C4 work | Fix the drift first |
| Which model is the instrument for the Sobol and teleconnection work? | 2.4, 2.7 | The best-calibrated model, with the noise model as structural uncertainty | — |

---

## 7. Order of work

1. **Now, no SDL checkpoint needed (local):** ✅ done 2026-11-06, except the fixer-correction logging (1.3) and the
   long-run spectra (3.2), which belong with the one-step-portrait and long-run scripts (not written yet).
   - the start-perturbation preblock and the tendency-noise postblock, with tests, `credit check` and docs;
   - the new hindcast arms, including White land noise;
   - the scorer additions and output dressing;
   - a local test on 1981.
2. **When the epoch 11–20 job finishes and the Derecho clone is pulled (Casper):**
   - plain-CAMulator baselines first: hindcasts (1.2) and the long-run baseline (E9);
   - one-step portraits (1.1, 1.3), which are cheap and show whether SDL is under-dispersed or too additive
     before the hindcast GPU time is spent;
   - first-wave hindcasts in one batch (2.x);
   - long runs (3.x).
3. **After a first look:** second-wave arms, then the decision points (§6) choose the targeted studies.

### How this maps to the three papers (stochasticity path §6)

| Paper | Questions |
|---|---|
| **1.** *What the learned noise is* | 1.1–1.3, plus the one-step part of 2.3 |
| **2.** *Its value from weather to S2S* | 2.1–2.8 |
| **3.** *Whether it changes the emulated climate* | 3.1–3.3, 4.1 |

---

## 8. Risks and caveats

- **Small S2S differences, limited samples.** 99 starts give modest power. Use paired comparisons and block
  bootstrap, and expect calibration results to be clearer than skill results.
- **Two held-out years.** At most one ENSO cycle, so 2.7 is tentative. ☐ Are other held-out CESM years or
  members available?
- **Unequal amplitudes.** Noise models are sized differently. Report each both as designed and at matched early
  spread where it matters (2.2).
- **Tendency noise perturbs the dynamics.** The emulator's 6-h change includes advection. The thermodynamic-only
  variant and 1.3 check how much this matters.
- **The fixers can hide unphysical noise.** 1.3 measures how much they correct.
- **SST is prescribed.** ENSO is a forcing, not a mode; frame results as atmospheric-bridge effects.
- **Long-run comparisons use training years.** Fine for statistics, not for trajectories.

---

## 9. Open questions

1. ☐ Are there held-out CESM years or ensemble members beyond 2013–14? This matters most for 2.7 and the skill
   questions.
2. ☐ Is there a CAM/CESM initial-condition ensemble with the same SST, for the saturation spread (2.6)?
3. ☐ Multiplicative noise: start with tendency noise (inference-only), and defer a trained Multiplicative SDL
   until 1.1 and 2.3 justify it?
4. ☐ Start-perturbation method (§2.1).
5. ☐ Long runs: which forcing years, and how many members per model?
6. ☐ Theme priority if GPU time is tight: S2S (theme 2) or climate (theme 3) first?
