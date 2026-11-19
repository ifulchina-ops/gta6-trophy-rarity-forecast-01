# Methodology — Week-One Rarity Forecaster

## What the program does

For each of the 55 GTA VI trophies it forecasts the percent of Exophase-tracked
players who will have earned it at the case's target, as five quantiles
(0.05 / 0.25 / 0.5 / 0.75 / 0.95). The output is `<id>.out`:
`{"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}, ...]}`.

## The model

An Exophase rarity is a ratio: `r = 100 * E / N`, earners over tracked players.
Both sides move over the horizon, so the forecast models both:

```
E_T = E_last * G + m * (N_T - N_last)
r_T = 100 * E_T / N_T
```

- `E_last`, `N_last` — earners and tracked players in the last read of the case.
- `N_T` — tracked players at the target, projected from the history as a scenario
  grid (not a point). Two regimes:
  - *Identified growth* (log-range of tracked players >= 0.7): log-linear OLS fit
    of `ln N` on time, projected to the target, centre capped at 300x the last
    read; five scenarios at z = -1.5, -0.75, 0, 0.75, 1.5 with spread
    `sigma_N = 0.35 + 0.30 * sqrt(horizon_days)`.
  - *Flat history* (the practice cases): no trend exists to fit. Exophase
    refreshes a game's counts on its own schedule; the program models refreshes
    as Poisson with one refresh per 4.5 hours, each multiplying the tracked
    population by 3x (k = 0..6 scenarios, weights renormalised).
- `m` — the marginal earn rate of newly tracked players. Fitted per trophy by
  OLS regression of earners on tracked players over the history (the slope is
  the joiners' earn rate), shrunk toward the kind prior with weight
  `R / (R + 1)`, `R` = log-range of tracked players. Flat history gives weight
  0: pure prior.
- `G = 1 + g_kind * horizon_days`, capped at 2.5 — old-cohort earning over the
  horizon (early-story trophies are earned fast by players already tracked;
  the platinum / 100% barely move).
- The kind prior's marginal rate is tenure-scaled: joiners arrive uniformly
  over the horizon, so the average joiner has half the horizon to play:
  `s = S * (1 - exp(-(horizon_days / 2) / tau))`. One exception, driven by the
  data itself: a trophy already held by >= 99% of tracked players gets
  `s = 0.98` — "Complete Gone Fishing" sat at 100% before and after the
  launch-night refresh, so its joiner rate is ~1, not the tenure curve's value.
- Quantiles come from the scenario mixture: each scenario contributes a
  logit-normal with spread 0.30, and the five levels are read off the mixture
  CDF on a fixed 2401-point logit grid. As `N_T` grows large, `r_T` converges
  to `m`, so a mis-projected population barely moves trophies whose marginal
  rate the history identifies.

## Kind priors (all estimated — see below)

| Kind | S (week-one steady state) | tau (days) | g (/day) |
|---|---|---|---|
| tutorial (Gone Fishing) | 0.97 | 0.1 | 0 |
| chapter 1 / 2 / 3 / 4 | 0.32 / 0.22 / 0.15 / 0.10 | 1.2 / 1.8 / 2.4 / 3.0 | 0.25 / 0.20 / 0.15 / 0.12 |
| finale (final mission, gold) | 0.05 | 4.0 | 0.08 |
| story (named mission) | 0.12 | 2.0 | 0.15 |
| strand (final mission for X) | 0.05 | 3.0 | 0.10 |
| grind (counts: 250 stunts, 20 races, …) | 0.03, scaled down with the count | 5.0 | 0.05 |
| collectible | 0.02 | 5.0 | 0.03 |
| skill (gold medals, races at every location) | 0.04 | 3.5 | 0.08 |
| event (one-off feats) | 0.08 | 2.0 | 0.10 |
| completionist (platinum / 100%) | 0.002 | 10 | 0.02 |
| vague (ambiguous secret trophies) | 0.03 | 3.0 | 0.08 |

Kinds are classified at run time from the trophy's official name, description
and tier in the case file — no trophy list is hardcoded. The program therefore
works unchanged on the final case, whose 55 trophies are the same list.

## Confirmed vs estimated

**CONFIRMED** (from the competition kit and the case files themselves):

- The 55 trophies, their tiers (1 platinum, 2 gold, 6 silver, 46 bronze), names
  and descriptions — the case file, cut from Exophase's public trophy page
  (`https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/`).
- The launch-night reads: 638 players tracked, every rate unchanged from the
  first probe (2026-11-18 23:50Z) through the 04:00Z read, then a refresh
  between 04:00Z and 04:30Z to 3,868 players and 4,276 trophies earned, with
  every rate but the one at 100% falling (kit's `competition.yaml`).
- The scoring rule: pinball loss on the logit scale over five quantiles,
  rates clamped to [0.01, 99.99] before the logit; 8,000 ms per case, 1 CPU,
  1,024 MB.

**ESTIMATED** (reasoned, not official — no official GTA VI data exists yet):

- The kind priors `S / tau / g`. Basis: GTA V / GTA Online lifetime rarity
  curves (early story trophies reach 60–80% lifetime, late story 30–50%,
  100% completion ~1–2%, long grinds and collectible sets 2–10% lifetime),
  discounted hard for week one: launch-week players average a few days of
  play, so week-one rates sit far below lifetime rates, and story trophies
  early in the game are earned far faster than late-game or grind trophies.
- The refresh cadence (~4.5 h) and size (~3x central). Basis: the single
  observed gap (flat 23:50Z→04:00Z, refreshed by 04:30Z) and the one observed
  jump (638 → 3,868 players, ~6x). The 3x centre is deliberately below the
  single 6x observation: a first-refresh backlog is expected to be larger
  than a typical refresh.
- The spreads (0.30 within-scenario, `sigma_N` for the population projection).
  Chosen so the 90% band covers the refresh-vs-no-refresh ambiguity on a flat
  history and the extrapolation error on a 4.5-day horizon.

No "official" GTA VI numbers are claimed anywhere in the entry. Nothing is
copied from another guide or entry: the trophy data comes from the case file,
and every line of code and prose in this repository was written for this entry.

## Validation done locally

- Practice cases p00/p01: run in ~0.3 s each (cap 8 s), byte-identical across
  two runs, 55 valid rows each, output 2,792 bytes (cap 65,536).
- Against a pure-dilution proxy answer for p00 (earners unchanged, players
  638 → 3,868, tutorial stays 100%): pinball loss 1.06 vs the house
  persistence baseline's 1.69 (typical miss 0.61 pp vs 0.73 pp). The proxy is
  an extreme (new joiners earn nothing beyond the tutorial); the real answer
  will sit between pure dilution and persistence, and the model's band covers
  both.
- A synthetic final-like case (30 reads, players 638 → ~42,000 over 2.5 days,
  4.5-day horizon): the fitted branch engages, chapter 1 forecasts ~44%
  (from 47.5% last, converging toward its fitted joiner rate), the platinum
  ~0.09%, the tutorial ~98% — the shape the model is designed to produce.
