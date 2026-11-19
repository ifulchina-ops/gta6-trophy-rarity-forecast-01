# Week-One Rarity Forecaster

[![gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ifulchina-ops/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ifulchina-ops/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

A deterministic forecaster for all **55 Grand Theft Auto VI trophies** (PlayStation 5): for each trophy it predicts the percent of the players Exophase tracks who will have earned it one week after launch, as five quantiles (5% / 25% / 50% / 75% / 95%). It reads the house's recording of Exophase's public trophy page — the case file — and nothing else: no network, no clock, no randomness.

The core idea is that an Exophase rarity is a ratio, **earners / tracked players**, and both sides move. When the tracked population grows, rates are diluted toward the earn rate of the newly tracked players; meanwhile the players already tracked keep earning. The program fits that dilution from the case's own history where the history identifies it (a log-linear projection of the tracked population, plus a per-trophy regression of earners on players-tracked for the joiners' earn rate), and falls back to reasoned kind-based priors where the history is flat — as it is on launch night, when every read shows the same 638 players and then one refresh jumps to 3,868. Trophy kinds come from the official descriptions in the case file itself: chapter trophies, the finale, 100% completion, grinds, collectibles, skill tests and one-off events each get their own steady-state rate and time-to-earn scale. Nothing about GTA VI's future is presented as official data: what is confirmed (the 55 trophies, the launch-night reads, the refresh) and what is estimated (every behavioural prior) is labelled in `METHODOLOGY.md`.

On the practice cases the model beats the house persistence baseline on the refresh-realised proxy answer (pinball loss 1.06 vs 1.69), because its lower quantiles already expect the refresh that persistence rules out.

## How the forecast behaves

- **Flat history (launch night).** No trend can be fitted, so the tracked population is modelled as a refresh process: about one refresh every 4.5 hours, each multiplying the tracked population ~3x. The no-refresh scenario still holds most of the mass, so the median stays near the last rate — but the lower quantiles reach deep into the refresh scenarios, and the tutorial trophy ("Complete Gone Fishing", the only one at 100%) is forecast to stay near 100% because every tracked player holds it before and after the refresh.
- **Growing history (the final case).** The log-linear trend of the tracked population and the per-trophy earners-on-players regression take over, shrunk toward the kind priors with weight R / (R + 1) where R is the log-range of the tracked population. As the projected population grows large, each trophy's rate converges to its joiners' earn rate — so a mis-projected population barely moves trophies whose marginal rate the history identifies.
- **Uncertainty is honest.** The five quantiles come from a scenario mixture, not from a single spread: a flat history gives a wide, skewed band; a history with real growth gives a tighter, trend-following one.

## Run it

Node 22, standard library only, no build step:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Each `<id>.in` in the input directory produces `<id>.out` in the output directory: `{"trophies":[{"id":1,"q":[q05,q25,q50,q75,q95]}, ...]}`, one row per trophy, five finite numbers from 0 to 100, non-decreasing. Two runs on the same input write byte-identical output, in about 0.3 s per case — far inside the grader's 8 s per-case cap.

## Files

- `main.ts` — the forecaster (the entrypoint)
- `METHODOLOGY.md` — the model, its constants, what is confirmed vs estimated, and the sources
- `LICENSE` — 0BSD

## License

0BSD — see [LICENSE](LICENSE).

---

Built for [make.fun](https://make.fun) — [gta6-trophy-rarity-forecast-01: Forecast how many GTA VI players earn each trophy in week one](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).
