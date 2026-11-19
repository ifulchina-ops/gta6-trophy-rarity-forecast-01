// Week-One Rarity Forecaster — gta6-trophy-rarity-forecast-01 (program/v1, Node 22, stdlib only).
//
// Forecasts, for every Grand Theft Auto VI trophy on PlayStation 5, the percent of the players
// Exophase tracks for the game who will have earned it at the case's target, as five quantiles
// (0.05 / 0.25 / 0.5 / 0.75 / 0.95). Reads every <id>.in in the input directory (the grader runs
// one case at a time) and writes <id>.out: {"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}, ...]}.
// No network, no clock, no randomness: the same case file always produces the same bytes.
//
// MODEL — a dilution + earning mixture, fitted from the case's own history where the history
// identifies it, with kind-based priors where it does not. See METHODOLOGY.md for the full
// write-up, the confirmed-vs-estimated labels, and the sources.
//
//   1. Exophase's rate is earners / tracked players. When the tracked population grows, rates
//      are diluted toward the earn rate of the marginal (newly tracked) player; meanwhile the
//      players already tracked keep earning. Per trophy:
//        E_T = E_last * G + m * (N_T - N_last)
//        r_T = 100 * E_T / N_T
//      E_last = earners in the last read, N_last = players tracked then, N_T = players tracked
//      at the target, G = old-cohort growth over the horizon, m = marginal earn rate of joiners.
//   2. N_T is projected from the history: a log-linear fit of players-tracked on time when the
//      history shows real growth (log-range >= 0.7), else a refresh-process prior (Exophase
//      refreshes a game's counts roughly every 4-5 hours; each refresh multiplies the tracked
//      population — the one refresh the house recorded, launch night, was ~6x). Either way N_T
//      is a small scenario grid, not a point: the forecast is the mixture over the scenarios.
//   3. m is fitted per trophy by regressing earners on players-tracked over the history
//      (the slope is the joiners' earn rate), shrunk toward a kind prior with weight
//      R / (R + 1), R = log-range of players-tracked. Flat history (the practice cases) gives
//      weight 0: pure prior. The prior is tenure-scaled: a joiner who arrives mid-horizon has
//      on average half the horizon to play, so s = S * (1 - exp(-(horizon/2) / tau)) where S is
//      the trophy kind's week-one steady-state rate and tau its time-to-earn scale.
//   4. G = 1 + g_kind * horizon_days (capped at 2.5): early-story trophies are earned fast by
//      players already tracked; the platinum / 100% barely move.
//   5. Quantiles come from the scenario mixture: each scenario contributes a logit-normal with
//      spread 0.30, and the five levels are read off the mixture CDF on a fixed logit grid.
//      A flat history therefore yields a wide, honest spread (the no-refresh scenario holds
//      most of the mass, so the median stays near the last rate while q05 reaches deep into
//      the refresh scenarios); a history with real growth yields a tighter, trend-following
//      forecast. As N_T grows large the rate converges to m, so a mis-projected N_T barely
//      moves trophies whose marginal rate is identified — the dilution error washes out.
//
// WHAT IS CONFIRMED vs ESTIMATED (see METHODOLOGY.md for sources):
//   CONFIRMED: the 55 trophies, their tiers and descriptions (the case file itself, cut from
//   Exophase's public trophy page); the launch-night reads (638 players, rates flat, then a
//   refresh to 3,868 players with every rate but the 100% one falling — the kit's
//   competition.yaml); the scoring rule (pinball loss on the logit scale, 5 quantiles).
//   ESTIMATED: everything about player behaviour — the kind priors S / tau / g, the refresh
//   cadence (~4.5 h) and size (~3x central), the spreads. They are reasoned from GTA V /
//   GTA Online lifetime rarity curves and launch-week play patterns, not from any official
//   GTA VI data (none exists yet). No "official" GTA VI numbers are claimed anywhere.
//
// Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for
// /input and /output:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

/** Within-scenario spread of a rate on the logit scale. */
const SIGMA_WITHIN = 0.3;
/** Exophase refreshes a game's counts about this often (hours); see METHODOLOGY.md. */
const REFRESH_EVERY_HOURS = 4.5;
/** Central growth of the tracked population per refresh (multiplier). */
const REFRESH_GROWTH = 3;
/** Hard cap on the projected tracked population, as a multiple of the last read. */
const MAX_GROWTH_MULTIPLE = 300;

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}

interface Read {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { name: string; platform: string; release_date: string; source: string };
  trophies: Trophy[];
  history: Read[];
  notes: string;
}

type Kind =
  | 'tutorial'
  | 'chapter1'
  | 'chapter2'
  | 'chapter3'
  | 'chapter4'
  | 'finale'
  | 'story'
  | 'strand'
  | 'grind'
  | 'collectible'
  | 'skill'
  | 'event'
  | 'completionist'
  | 'vague';

/** Kind prior: S = week-one steady-state earn rate among players with a week of tenure;
 *  tau = days to reach it; g = old-cohort earners' growth per day of horizon. */
const KIND_PRIOR: Record<Kind, { s: number; tau: number; g: number }> = {
  tutorial: { s: 0.97, tau: 0.1, g: 0 },
  chapter1: { s: 0.32, tau: 1.2, g: 0.25 },
  chapter2: { s: 0.22, tau: 1.8, g: 0.2 },
  chapter3: { s: 0.15, tau: 2.4, g: 0.15 },
  chapter4: { s: 0.1, tau: 3.0, g: 0.12 },
  finale: { s: 0.05, tau: 4.0, g: 0.08 },
  story: { s: 0.12, tau: 2.0, g: 0.15 },
  strand: { s: 0.05, tau: 3.0, g: 0.1 },
  grind: { s: 0.03, tau: 5.0, g: 0.05 },
  collectible: { s: 0.02, tau: 5.0, g: 0.03 },
  skill: { s: 0.04, tau: 3.5, g: 0.08 },
  event: { s: 0.08, tau: 2.0, g: 0.1 },
  completionist: { s: 0.002, tau: 10, g: 0.02 },
  vague: { s: 0.03, tau: 3.0, g: 0.08 },
};

const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
const clampRate = (x: number): number => clamp(x, 0.01, 99.99);
const logit = (pct: number): number => Math.log(clampRate(pct) / (100 - clampRate(pct)));
const invLogitPct = (x: number): number => 100 / (1 + Math.exp(-x));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

/** Standard normal CDF (Abramowitz & Stegun 7.1.26, |error| < 1.5e-7). */
function normCdf(x: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const poly =
    t *
    (0.31938153 +
      t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  const p = 1 - Math.exp(-(x * x) / 2) / Math.sqrt(2 * Math.PI) * poly;
  return x >= 0 ? p : 1 - p;
}

/** Least-squares line y ~ a + b*x; null when x has no variance or fewer than 2 points. */
function ols(xs: number[], ys: number[]): { a: number; b: number } | null {
  if (xs.length < 2) return null;
  const n = xs.length;
  const mx = xs.reduce((s, v) => s + v, 0) / n;
  const my = ys.reduce((s, v) => s + v, 0) / n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  if (sxx <= 0) return null;
  const b = sxy / sxx;
  return { a: my - b * mx, b };
}

/** Classify a trophy from its official description and tier. No trophy list is hardcoded:
 *  the case file carries the names and descriptions; the rules below read them. */
function classify(t: Trophy): { kind: Kind; count: number } {
  const text = `${t.name} ${t.description}`;
  const lower = text.toLowerCase();
  if (/gone fishing/i.test(text)) return { kind: 'tutorial', count: 1 };
  const chapter = /complete chapter (\d)/i.exec(text);
  if (chapter) {
    const n = Number(chapter[1]);
    const kind =
      n <= 1 ? 'chapter1' : n === 2 ? 'chapter2' : n === 3 ? 'chapter3' : 'chapter4';
    return { kind, count: n };
  }
  if (/100% completion/i.test(text) || t.tier === 'platinum')
    return { kind: 'completionist', count: 1 };
  if (/complete the final mission\.?$/i.test(t.description) && t.tier === 'gold')
    return { kind: 'finale', count: 1 };
  if (/complete the final mission for/i.test(text)) return { kind: 'strand', count: 1 };
  const countMatch = /(\d+)\s*(stunts|races|locations|challenges|vehicles|bank vaults|shops|card sets|spots|properties|missions)/i.exec(
    text,
  );
  const count = countMatch ? Number(countMatch[1]) : 1;
  if (/stunts|different races|locations|challenges|card sets|bank vaults|vehicles|properties/i.test(text))
    return { kind: 'grind', count };
  if (/meditate at every spot|explore all|hunt a unique|collectible/i.test(text))
    return { kind: 'collectible', count };
  if (/gold medal|win .* race at every|melee drive-by/i.test(text))
    return { kind: 'skill', count };
  if (/complete /i.test(text)) return { kind: 'story', count: 1 };
  if (
    /lose a .* wanted|rob |take down \d+ enemies|unscheduled departure|enjoy a cocktail|follow-through|spend \$|earn \$|purchase all|register \d+/i.test(
      text,
    )
  )
    return { kind: 'event', count: 1 };
  return { kind: 'vague', count: 1 };
}

/** The kind prior's marginal earn rate for joiners over this horizon. Joiners arrive
 *  uniformly over the horizon, so the average joiner has half the horizon to play. */
function priorMarginal(
  kind: Kind,
  count: number,
  horizonDays: number,
  lastRate: number,
): number {
  // A trophy every tracked player already holds stays held by joiners: Gone Fishing sat at
  // 100% before and after the launch-night refresh, so its joiner rate is ~1, not the
  // tenure curve's value.
  if (lastRate >= 99) return 0.98;
  const p = KIND_PRIOR[kind];
  let s = p.s;
  // Long counts are earned slower: scale the steady state down with the count.
  if (kind === 'grind') s = p.s * Math.sqrt(30 / (30 + count));
  const tenure = horizonDays / 2;
  return clamp(s * (1 - Math.exp(-tenure / p.tau)), 0, 0.99);
}

/** Scenario grid for the tracked population at the target: {n, weight} pairs. */
function playerScenarios(
  history: Read[],
  horizonHours: number,
  horizonDays: number,
): Array<{ n: number; w: number }> {
  const usable = history.filter((r) => r.players_tracked > 0);
  const nLast = usable.length > 0 ? usable[usable.length - 1].players_tracked : 0;
  if (nLast <= 0) return [{ n: 1, w: 1 }];
  const ns = usable.map((r) => r.players_tracked);
  const nMax = Math.max(...ns);
  const nMin = Math.min(...ns);
  const logRange = Math.log(nMax / nMin);
  const t0 = Date.parse(usable[0].at);
  const fit = ols(
    usable.map((r) => (Date.parse(r.at) - t0) / 3_600_000),
    usable.map((r) => Math.log(r.players_tracked)),
  );

  if (fit !== null && logRange >= 0.7) {
    // Identified growth: project the log-linear trend, with a wide spread for a long
    // extrapolation, and cap the centre at 300x the last read.
    const sigmaN = 0.35 + 0.3 * Math.sqrt(horizonDays);
    const centre = clamp(
      nLast * Math.exp(fit.b * horizonHours),
      nLast,
      nLast * MAX_GROWTH_MULTIPLE,
    );
    const zs = [-1.5, -0.75, 0, 0.75, 1.5];
    const ws = [0.1028, 0.239, 0.3166, 0.239, 0.1028];
    return zs.map((z, i) => ({
      n: clamp(centre * Math.exp(z * sigmaN), nLast, nLast * MAX_GROWTH_MULTIPLE),
      w: ws[i],
    }));
  }

  // Flat history: no trend to fit. Exophase refreshes roughly every 4.5 hours; each
  // refresh multiplies the tracked population. k refreshes ~ Poisson(lambda).
  const lambda = Math.min(horizonHours / REFRESH_EVERY_HOURS, 2.5);
  const out: Array<{ n: number; w: number }> = [];
  let total = 0;
  for (let k = 0; k <= 6; k++) {
    // Poisson pmf, computed iteratively to stay exact and deterministic.
    let pmf = Math.exp(-lambda);
    for (let j = 1; j <= k; j++) pmf *= lambda / j;
    out.push({ n: clamp(nLast * Math.pow(REFRESH_GROWTH, k), nLast, nLast * MAX_GROWTH_MULTIPLE), w: pmf });
    total += pmf;
  }
  return out.map((s) => ({ ...s, w: s.w / total }));
}

/** Forecast one trophy's five quantiles (percent, non-decreasing, 4 decimals). */
function forecastTrophy(
  t: Trophy,
  idx: number,
  c: Case,
  scenarios: Array<{ n: number; w: number }>,
  logRange: number,
): number[] {
  const horizonDays = c.horizon_minutes / 1440;
  const { kind, count } = classify(t);
  const prior = KIND_PRIOR[kind];
  const g = Math.min(1 + prior.g * horizonDays, 2.5);

  // Last read that lists this trophy, and its earners.
  let lastRate: number | null = null;
  let lastN = 0;
  for (let i = c.history.length - 1; i >= 0; i--) {
    const r = c.history[i].rates[idx];
    if (r !== null && r !== undefined && c.history[i].players_tracked > 0) {
      lastRate = r;
      lastN = c.history[i].players_tracked;
      break;
    }
  }
  if (lastRate === null) {
    // Never observed: fall back to the prior alone, centred on the steady state.
    const s = priorMarginal(kind, count, horizonDays, 0);
    const centre = clampRate(100 * s);
    return Z.map((z) => round4(clampRate(invLogitPct(logit(centre) + z * 0.8))));
  }
  const eLast = (lastRate / 100) * lastN;

  // Fitted marginal earn rate: regress earners on players-tracked; shrink to the prior.
  const sPrior = priorMarginal(kind, count, horizonDays, lastRate);
  let m = sPrior;
  const xs: number[] = [];
  const ys: number[] = [];
  for (const r of c.history) {
    const rate = r.rates[idx];
    if (rate !== null && rate !== undefined && r.players_tracked > 0) {
      xs.push(r.players_tracked);
      ys.push((rate / 100) * r.players_tracked);
    }
  }
  if (new Set(xs).size >= 2) {
    const fit = ols(xs, ys);
    if (fit !== null) {
      const wFit = logRange / (logRange + 1);
      m = clamp(wFit * fit.b + (1 - wFit) * sPrior, 0, 0.99);
    }
  }

  // Scenario rates, then the mixture CDF on a fixed logit grid.
  const mus = scenarios.map((s) => {
    const eT = eLast * g + m * Math.max(s.n - lastN, 0);
    return logit(clampRate((100 * eT) / s.n));
  });
  const grid: number[] = [];
  const lo = logit(0.01);
  const hi = logit(99.99);
  const steps = 2400;
  for (let i = 0; i <= steps; i++) grid.push(lo + ((hi - lo) * i) / steps);
  const cdf = grid.map((x) => {
    let p = 0;
    for (let s = 0; s < scenarios.length; s++) {
      p += scenarios[s].w * normCdf((x - mus[s]) / SIGMA_WITHIN);
    }
    return p;
  });
  const qs = LEVELS.map((tau) => {
    let j = cdf.findIndex((p) => p >= tau);
    if (j < 0) j = grid.length - 1;
    return round4(clampRate(invLogitPct(grid[j])));
  });
  // Enforce non-decreasing after rounding.
  for (let i = 1; i < qs.length; i++) qs[i] = Math.max(qs[i], qs[i - 1]);
  return qs;
}

function forecastCase(c: Case): { trophies: Array<{ id: number; q: number[] }> } {
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j]))
    throw new Error(`${c.id} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  if (c.history.length === 0) throw new Error(`${c.id} has no history`);
  const horizonHours = c.horizon_minutes / 60;
  const horizonDays = c.horizon_minutes / 1440;
  const scenarios = playerScenarios(c.history, horizonHours, horizonDays);
  const ns = c.history.filter((r) => r.players_tracked > 0).map((r) => r.players_tracked);
  const logRange = ns.length > 0 ? Math.log(Math.max(...ns) / Math.min(...ns)) : 0;
  const trophies = c.trophies.map((t, i) => ({
    id: t.id,
    q: forecastTrophy(t, i, c, scenarios, logRange),
  }));
  return { trophies };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith('.in'))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const out = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies: out.trophies })}\n`);
}
