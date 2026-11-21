# Week-One Rarity Forecaster

[![gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ifulchina-ops/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ifulchina-ops/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

A deterministic forecaster for all **55 Grand Theft Auto VI trophies** (PlayStation 5): for each trophy it predicts the percent of the players Exophase tracks who will have earned it one week after launch, as five quantiles (5% / 25% / 50% / 75% / 95%). It reads the house's recording of Exophase's public trophy page — the case file — and nothing else: no network, no clock, no randomness.

The core idea is that an Exophase rarity is a ratio, **earners / tracked players**, and both sides move. When the tracked population grows, rates are diluted toward the earn rate of the newly tracked players; meanwhile the players already tracked keep earning. R2 rebuilds R1's dilution model around four fixes: refresh timing is staleness-aware (a page stale for hours is due a refresh soon, so the practice median expects the launch-night refresh instead of staying near the last rate); the marginal joiner rate is fitted from refresh jumps, not a levels regression; old-cohort earning is fitted from jump residuals, not a pure prior; and story progress follows a delayed sigmoid in tenure from launch, not a linear curve (a player 90 minutes in has finished nothing story-shaped yet). Trophy kinds come from the official descriptions in the case file itself. Nothing about GTA VI's future is presented as official data: what is confirmed (the 55 trophies, the launch-night reads, the refresh) and what is estimated (every behavioural prior) is labelled in `METHODOLOGY.md`.

On the practice cases the model scores a pinball loss of ~0.20 against the refresh-realised proxy answers (R1: 0.897; house persistence baseline: ~1.51), and its median total (4,265 trophies earned) matches the known refresh total (4,276) within 0.3%. On synthetic final-like cases with known ground truth it scores 0.13–0.15, against 0.29–0.33 for persistence and 1.70–1.73 for dilution alone.

## How the forecast behaves

- **Flat history (launch night).** No trend can be fitted, so the tracked population is modelled as a refresh process on a renewal schedule (~4.5 h cadence, anchored at the last observed refresh, with the last-refresh time integrated over one cadence back when no jump was ever observed); each refresh multiplies the tracked population by a lognormal size whose median scales with staleness. The median expects the refresh; the upper quantiles still allow for no refresh. The tutorial trophy ("Complete Gone Fishing", the only one at 100%) is forecast to stay near 100% because every tracked player holds it before and after the refresh.
- **Growing history (the final case).** A recency-weighted log-linear trend projects the tracked population, and each trophy's marginal joiner rate and old-cohort earning rate are fitted from the refresh jumps (dE = m·dN + r·E·dt), recency-weighted and shrunk toward the kind priors. As the projected population grows large, each trophy's rate converges to its joiners' earn rate — so a mis-projected population barely moves trophies whose marginal rate the history identifies.
- **Uncertainty is honest.** The five quantiles come from a scenario mixture, not from a single spread: a flat history gives a wide, skewed band; a history with real growth gives a tighter, trend-following one.

## Run it

Node 22, standard library only, no build step:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Each `<id>.in` in the input directory produces `<id>.out` in the output directory: `{"trophies":[{"id":1,"q":[q05,q25,q50,q75,q95]}, ...]}`, one row per trophy, five finite numbers from 0 to 100, non-decreasing. Two runs on the same input write byte-identical output, in about 0.5 s per case — far inside the grader's 8 s per-case cap.

## Files

- `main.ts` — the forecaster (the entrypoint)
- `METHODOLOGY.md` — the model, its constants, what is confirmed vs estimated, and the sources
- `LICENSE` — 0BSD

## License

0BSD — see [LICENSE](LICENSE).

---

Built for [make.fun](https://make.fun) — [gta6-trophy-rarity-forecast-01: Forecast how many GTA VI players earn each trophy in week one](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).
