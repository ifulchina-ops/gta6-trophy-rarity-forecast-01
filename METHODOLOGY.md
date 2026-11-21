# Methodology — Week-One Rarity Forecaster R2

## What the program does

For each of the 55 Grand Theft Auto VI trophies (PlayStation 5) it forecasts the percent
of the players Exophase tracks for the game who will have earned the trophy at the case's
target, as five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95). The input is the house's
recording of Exophase's public trophy page: a list of reads, each with a timestamp, the
number of players tracked, the total trophies earned, and the per-trophy rates. The
program reads the case file and nothing else: no network, no clock, no randomness.

## The model

An Exophase rarity is a ratio, earners / tracked players, and both sides move. Per trophy
the forecast is

    E_T = E_last · G + m · (N_T − N_last),        r_T = 100 · E_T / N_T

where E_last and N_last are the last read's earners and tracked players, G = 1 + r·horizon
(capped at 2.5) is the old cohort's growth, m is the marginal rate at which newly tracked
players hold the trophy, r is the old cohort's earning rate per day, and N_T is the
tracked population at the target. As N_T grows large the rate converges to m, so a
mis-projected N_T barely moves trophies whose marginal rate the history identifies.

### Tracked population at the target (N_T)

Two branches, chosen by the log-range of the tracked population in the history:

- **Growth branch** (log-range ≥ 0.7): a recency-weighted log-linear trend of log N on
  time (weights decay with a 24 h half-life, so recent growth dominates — launch-week
  inflow decays), projected to the target and capped at 300× the last read. Five
  scenarios on a z-grid (−1.5, −0.75, 0, 0.75, 1.5) with σ_N = 0.35 + 0.3·√days.
- **Flat branch** (log-range < 0.7): no trend can be fitted, so N_T comes from a refresh
  process. Refreshes recur on a renewal schedule with a lognormal cadence (median 4.5 h,
  σ 0.45), anchored at the last observed refresh (the last read where the tracked
  population jumped). When no jump was ever observed, the last refresh time is unknown
  — it predates the first read — so the staleness is integrated over one cadence back
  from the first read (five points, equal weight). Each refresh multiplies the tracked
  population by a lognormal size (σ 0.55) whose median scales with staleness,
  1 + staleness_hours (clamped to 1.5–10×): the longer the page sits stale, the bigger
  the backlog of unsynced players the next refresh picks up.

### Marginal rate (m) and old-cohort earning (r)

At each refresh the page jumps, and the jump identifies both quantities:

    ΔE = m·ΔN + r·E_prev·Δt

(joiners' holdings plus the old cohort's earning since the last refresh). R2 estimates
them in two passes over the jump intervals — consecutive reads where the tracked
population grew; flat stretches contribute nothing, because the page is frozen between
refreshes and a flat stretch shows no earning at all:

1. **m**: per jump, m_j = (ΔE − g_prior·E_prev·Δt) / ΔN (prior old-cohort earning
   subtracted), clamped to [0, 1], averaged with weights ΔN·recency. Shrunk toward the
   kind prior with weight totalJumpN / (totalJumpN + 2000 players).
2. **r**: per jump, r_j = (ΔE − m·ΔN) / (E_prev·Δt), clamped to [0, 2] per day, averaged
   with weights E_prev·Δt·recency (intervals with no prior earners carry no information
   about r). Shrunk toward the kind prior g with weight totalEarnerDays /
   (totalEarnerDays + 500 earner-days).

With no jumps both stay at their kind priors.

### Kind priors

Trophies are classified from their official name and description in the case file
(chapter trophies 1–4, finale, 100% completion/platinum, strands, grinds, collectibles,
skill tests, one-off events, vague/secret). Each kind has a steady-state week-one earn
rate S, a time scale tau (days to half of S), and an old-cohort growth rate g per day:

| kind | S | tau (days) | g (/day) |
|---|---|---|---|
| tutorial | 0.97 | 0.1 | 0 |
| chapter1 | 0.32 | 1.2 | 0.25 |
| chapter2 | 0.22 | 1.8 | 0.20 |
| chapter3 | 0.15 | 2.4 | 0.15 |
| chapter4 | 0.10 | 3.0 | 0.12 |
| finale | 0.05 | 4.0 | 0.08 |
| story | 0.12 | 2.0 | 0.15 |
| strand | 0.05 | 3.0 | 0.10 |
| grind | 0.03 | 5.0 | 0.05 |
| collectible | 0.02 | 5.0 | 0.03 |
| skill | 0.04 | 3.5 | 0.08 |
| event | 0.08 | 2.0 | 0.10 |
| completionist | 0.002 | 10 | 0.02 |
| vague | 0.03 | 3.0 | 0.08 |

Grinds scale S down with the count: S·√(30 / (30 + count)). A trophy every tracked
player already holds (last rate ≥ 99%) gets m = 0.98 regardless of kind — Gone Fishing
sat at 100% before and after the launch-night refresh.

The prior marginal rate follows a sigmoid in joiner tenure:

    m_prior = S · (t/tau)² / (1 + (t/tau)²)

where t is the joiner tenure: half the time from release to target, floored at half the
horizon and capped at the horizon. The sigmoid is ~quadratic at short tenures — a player
90 minutes into the game has finished nothing story-shaped yet — and saturates at S.

### Quantiles

Each N_T scenario gives a scenario rate and hence a logit centre μ. The forecast is the
scenario mixture of logit-normals (within-scenario spread σ = 0.25) evaluated on a fixed
2,401-point logit grid; the five quantiles are read off the mixture CDF and rounded to 4
decimals, then forced non-decreasing. A trophy never observed in the history falls back
to the kind prior alone, centred on the steady state with spread 0.8.

## What R2 changes from R1

R1 (loss 0.897 on practice) modelled refreshes as a memoryless Poisson process restarted
at the checkpoint, fitted m by a levels OLS of earners on players-tracked, used a pure
prior for old-cohort growth, and used a (1 − exp(−t/tau)) tenure curve with tenure =
horizon/2. R2 changes four things:

1. **Staleness-aware refresh timing.** R1 put ~72% of the mass on "no refresh" for the
   practice cases, leaving the median near the last rate. R2's renewal process anchors
   refreshes at the last observed refresh and integrates the unknown last-refresh time,
   so a page stale for hours expects its refresh inside the horizon.
2. **Jump-based m.** A levels OLS is dominated by the single largest (earliest, most
   hardcore) jump; the jump estimator tracks the marginal joiner at the target and is
   recency-weighted, since the marginal joiner gets more casual over launch week.
3. **Fitted r.** The jump residuals identify old-cohort earning per trophy; R2 fits it
   instead of trusting the prior alone.
4. **Delayed sigmoid tenure from launch.** R1's linear-at-small-t curve put the marginal
   rate of early-story trophies near 1.6% for a 90-minute horizon — over 50× the joiner
   rate the launch-night refresh actually showed (~0.03 trophies per joiner outside the
   tutorial). The sigmoid keeps short-tenure joiners near zero on story trophies.

## Validation

**Practice cases** (p00/p01). The true practice answers are the 04:30 refresh page
(3,868 players, 4,276 trophies earned), which is no longer fetchable; validation scores
the forecast against a range of plausible answers consistent with the known refresh
totals (pure dilution; the 103 extra non-tutorial earns spread ∝ old earners, ∝
√earners, or concentrated in the top 5 early trophies). R2 scores 0.18–0.27 across the
range (mean ~0.20), against R1's 0.897 and the house persistence baseline's ~1.51. The
model's median total (4,265 trophies earned at the refresh) matches the known total
(4,276) within 0.3%.

**Synthetic final-like cases.** Three generated cases with known ground truth (2.5 days
of step-function history, saturating player growth to ~500k, marginal joiners more
casual than the tracked population): R2 scores 0.13–0.15, against 0.29–0.33 for
persistence and 1.70–1.73 for dilution alone.

**Format and budget.** Output is 55 rows, five non-decreasing quantiles each, ~2.8 KB
(limit 65,536 bytes). Runtime is 0.3–1.1 s per case (limit 8 s). Two runs on the same
input write byte-identical output (no clock, no randomness, no network).

## Confirmed vs estimated

**Confirmed** (from the kit's competition.yaml and the case files themselves): the 55
trophies, their tiers, names and descriptions; the launch-night reads (638 players, rates
flat from 23:48, then a refresh to 3,868 players with every rate but the 100% one
falling); the refresh totals (943 → 4,276 trophies earned); the scoring rule (pinball
loss on the logit scale, five quantiles); the 8-second per-case budget and the 65,536-byte
output cap.

**Estimated** (reasoned, not official): every behavioural prior — the kind table, the
refresh cadence (~4.5 h) and size, the spreads, the shrinkage strengths. The kind priors
are reasoned from GTA V / GTA Online lifetime rarity curves and launch-week play
patterns; no official GTA VI numbers exist yet, and none are claimed. The refresh
cadence is inferred from the single observed launch-night gap (flat from 23:48, refreshed
by 04:30) plus the kit's "hours apart" note.

## Sources

- The competition kit's `competition.yaml` (rules, practice and final case definitions,
  the launch-night refresh narrative) and the case files `p00.in` / `p01.in`.
- The `README.md` of this repository for how to run the program.
