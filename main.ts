// Week-One Rarity Forecaster R2 — gta6-trophy-rarity-forecast-01 (program/v1, Node 22, stdlib only).
//
// Forecasts, for every Grand Theft Auto VI trophy on PlayStation 5, the percent of the players
// Exophase tracks for the game who will have earned it at the case's target, as five quantiles
// (0.05 / 0.25 / 0.5 / 0.75 / 0.95). Reads every <id>.in in the input directory (the grader runs
// one case at a time) and writes <id>.out: {"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}, ...]}.
// No network, no clock, no randomness: the same case file always produces the same bytes.
//
// MODEL — a dilution + earning mixture, fitted from the case's own history where the history
// identifies it, with kind-based priors where it does not. R2 rebuilds R1's model around four
// fixes (see METHODOLOGY.md for the full write-up and the validation):
//
//   1. REFRESH TIMING IS STALENESS-AWARE, not memoryless. Exophase refreshes a game's counts
//      on a quasi-periodic schedule (~4.5 h), so the time since the last refresh tells you
//      when the next one is due: a page stale for 3+ hours is likely to refresh inside a
//      90-minute horizon, while a freshly refreshed page is not. R1 modelled refreshes as a
//      Poisson process restarted at the checkpoint, which put ~72% of the mass on "no
//      refresh" for the practice cases and left the median near the last rate — the main
//      reason R1 scored 0.897 on practice while a dilution-centred forecast scores ~0.13.
//      R2 runs a renewal process over a cadence grid anchored at the last observed refresh.
//   2. THE MARGINAL RATE COMES FROM REFRESH JUMPS, not a levels regression. At each refresh
//      the page jumps: dE = m*dN + r*E_prev*dt — joiners' holdings plus the old cohort's
//      earning since the last refresh. R2 estimates m per jump (subtracting prior old-cohort
//      earning), recency-weights the jumps (the marginal joiner gets more casual over launch
//      week), and shrinks toward the kind prior by jump mass. A levels OLS is dominated by
//      the single largest (earliest, most hardcore) jump; the jump estimator tracks the
//      marginal joiner at the target.
//   3. OLD-COHORT EARNING IS FITTED FROM JUMP RESIDUALS, not a pure prior. Given m, each
//      jump's residual (dE - m*dN) over the inter-refresh gap identifies the old cohort's
//      earning rate r per trophy; R2 averages it recency-weighted and shrinks it to the
//      kind prior g. Flat stretches contribute nothing (the page is frozen between
//      refreshes, so a flat stretch shows no earning at all) — only jumps identify r.
//   4. JOINER TENURE COUNTS FROM LAUNCH, not from the horizon, AND STORY PROGRESS
//      IS DELAYED, not linear in tenure. A player who syncs at a refresh has already
//      been playing since they started — on launch night, hours — so the kind prior's
//      tenure curve uses (target - release)/2, floored at horizon/2 and capped at the
//      horizon, instead of horizon/2 always. And a player 90 minutes into the game has
//      finished nothing story-shaped yet: progress follows a sigmoid in tenure,
//      (t/tau)^2 / (1 + (t/tau)^2), which is ~quadratic at short tenures (you must play
//      a while before completing anything) and saturates at the steady state. R1's
//      (1 - exp(-t/tau)) is linear at small t and put the marginal rate of early-story
//      trophies near 1.6% for a 90-minute horizon — over 50x the joiner rate the
//      launch-night refresh actually showed (~0.03 trophies per joiner outside the
//      tutorial) — which is the second reason R1's medians stayed near the last rate.
//
//   Per trophy the forecast is still E_T = E_last * G + m * (N_T - N_last), r_T = 100*E_T/N_T,
//   with G = 1 + r*horizon_days (capped at 2.5) and N_T a scenario grid; quantiles come from
//   the scenario mixture on a fixed logit grid (within-scenario spread 0.25). As N_T grows
//   large the rate converges to m, so a mis-projected N_T barely moves trophies whose
//   marginal rate the history identifies.
//
// WHAT IS CONFIRMED vs ESTIMATED (see METHODOLOGY.md for sources):
//   CONFIRMED: the 55 trophies, their tiers and descriptions (the case file itself); the
//   launch-night reads (638 players, rates flat, then a refresh to 3,868 players with every
//   rate but the 100% one falling — the kit's competition.yaml); the scoring rule (pinball
//   loss on the logit scale, 5 quantiles); the refresh totals (943 -> 4,276 trophies earned).
//   ESTIMATED: everything about player behaviour — the kind priors S / tau / g, the refresh
//   cadence (~4.5 h) and size, the spreads. They are reasoned from GTA V / GTA Online
//   lifetime rarity curves and launch-week play patterns, not from any official GTA VI data
//   (none exists yet). No "official" GTA VI numbers are claimed anywhere.
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
const SIGMA_WITHIN = 0.25;
/** Exophase refresh cadence: median hours between refreshes of a game's counts. */
const REFRESH_CADENCE_MEDIAN_HOURS = 4.5;
/** Log-sigma of the refresh cadence (renewal-process grid). */
const REFRESH_CADENCE_SIGMA = 0.45;
/** Log-sigma of the refresh size multiplier. */
const REFRESH_SIZE_SIGMA = 0.55;
/** Hard cap on the projected tracked population, as a multiple of the last read. */
const MAX_GROWTH_MULTIPLE = 300;
/** Log-range of players-tracked above which the growth branch (trend fit) engages. */
const GROWTH_LOG_RANGE = 0.7;
/** Prior strength for the fitted marginal rate, in players of jump mass. */
const M_PRIOR_STRENGTH_PLAYERS = 2000;
/** Prior strength for the fitted earning rate, in earner-days of jump mass. */
const R_PRIOR_STRENGTH_EARNER_DAYS = 500;
/** Recency half-life for jump weighting (hours): jumps older than this count less. */
const JUMP_RECENCY_HALF_LIFE_HOURS = 24;

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

/** Lognormal density at x for median m and log-sigma s (unnormalised is fine here:
 *  callers renormalise over their grid). */
function lognormalWeight(x: number, median: number, sigma: number): number {
  const z = (Math.log(x) - Math.log(median)) / sigma;
  return Math.exp(-0.5 * z * z) / x;
}

/** Weighted least-squares line y ~ a + b*x; null when x has no variance or fewer
 *  than 2 points with positive weight. */
function wols(
  xs: number[],
  ys: number[],
  ws: number[],
): { a: number; b: number } | null {
  let sw = 0;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < xs.length; i++) {
    sw += ws[i];
    sx += ws[i] * xs[i];
    sy += ws[i] * ys[i];
  }
  if (sw <= 0 || xs.length < 2) return null;
  const mx = sx / sw;
  const my = sy / sw;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < xs.length; i++) {
    sxx += ws[i] * (xs[i] - mx) * (xs[i] - mx);
    sxy += ws[i] * (xs[i] - mx) * (ys[i] - my);
  }
  if (sxx <= 0) return null;
  const b = sxy / sxx;
  return { a: my - b * mx, b };
}

/** Classify a trophy from its official description and tier. No trophy list is hardcoded:
 *  the case file carries the names and descriptions; the rules below read them. */
function classify(t: Trophy): { kind: Kind; count: number } {
  const text = `${t.name} ${t.description}`;
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
  const countMatch =
    /(\d+)\s*(stunts|races|locations|challenges|vehicles|bank vaults|shops|card sets|spots|properties|missions)/i.exec(
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

/** The kind prior's marginal earn rate for joiners. Joiners at a refresh have already
 *  been playing since they started — on launch night, hours — so tenure runs from the
 *  release, not from the horizon: the caller passes tenureDays (already floored at
 *  horizon/2 and capped at the horizon). Progress is a sigmoid in tenure,
 *  (t/tau)^2 / (1 + (t/tau)^2): ~quadratic at short tenures (a player 90 minutes in has
 *  finished nothing story-shaped yet) and saturating at the steady state s. */
function priorMarginal(
  kind: Kind,
  count: number,
  tenureDays: number,
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
  const x = tenureDays / p.tau;
  return clamp(s * (x * x) / (1 + x * x), 0, 0.99);
}

/** Joiner tenure in days: how long the average player syncing near the target has been
 *  playing. Half the time from release to target, floored at half the horizon (a joiner
 *  arriving during the horizon has at least that long) and capped at the horizon. */
function joinerTenureDays(c: Case, horizonDays: number): number {
  const releaseMs = Date.parse(`${c.game.release_date}T00:00:00Z`);
  const targetMs = Date.parse(c.target);
  if (Number.isNaN(releaseMs) || Number.isNaN(targetMs) || targetMs <= releaseMs)
    return horizonDays / 2;
  const sinceReleaseDays = (targetMs - releaseMs) / 86_400_000;
  return clamp(sinceReleaseDays / 2, horizonDays / 2, horizonDays);
}

/** When the tracked population last jumped (a refresh), and how stale the page is at the
 *  checkpoint. With no jump ever observed, the last refresh is UNKNOWN — it happened
 *  before the first read, possibly much before — so the staleness from the first read is
 *  only a lower bound; the caller integrates over the uncertainty (one cadence back). */
function lastRefreshInfo(
  history: Read[],
  checkpointMs: number,
): { lastRefreshMs: number; stalenessHours: number; hadJump: boolean } {
  const usable = history.filter((r) => r.players_tracked > 0);
  let lastRefreshMs = usable.length > 0 ? Date.parse(usable[0].at) : checkpointMs;
  let hadJump = false;
  for (let i = 1; i < usable.length; i++) {
    if (usable[i].players_tracked !== usable[i - 1].players_tracked) {
      lastRefreshMs = Date.parse(usable[i].at);
      hadJump = true;
    }
  }
  return {
    lastRefreshMs,
    stalenessHours: Math.max((checkpointMs - lastRefreshMs) / 3_600_000, 0),
    hadJump,
  };
}

/** Distribution of the number of refreshes inside the horizon, from a renewal process:
 *  refreshes recur every cadence hours (lognormal around ~4.5 h), anchored at the last
 *  observed refresh. A stale page is due a refresh soon; a fresh page is not. Returns
 *  [{k, w}] for k = 0..6, weights normalised. */
function refreshCountDistribution(
  stalenessHours: number,
  horizonHours: number,
): Array<{ k: number; w: number }> {
  const cadences = [2, 3, 4, 4.5, 5, 6, 8];
  const counts = new Map<number, number>();
  let total = 0;
  for (const c of cadences) {
    const w = lognormalWeight(c, REFRESH_CADENCE_MEDIAN_HOURS, REFRESH_CADENCE_SIGMA);
    // Time to the next refresh: the cadence minus the staleness; an overdue refresh
    // (cadence shorter than the staleness) is due imminently.
    let next = c - stalenessHours;
    if (next < 0.2) next = 0.2;
    let k = 0;
    for (let t = next; t <= horizonHours && k <= 6; t += c) k++;
    counts.set(k, (counts.get(k) ?? 0) + w);
    total += w;
  }
  const out: Array<{ k: number; w: number }> = [];
  for (const [k, w] of [...counts.entries()].sort((a, b) => a[0] - b[0]))
    out.push({ k, w: w / total });
  return out;
}

/** Scenario grid for the tracked population at the target: {n, weight} pairs. */
function playerScenarios(
  history: Read[],
  horizonHours: number,
  horizonDays: number,
  checkpointMs: number,
): Array<{ n: number; w: number }> {
  const usable = history.filter((r) => r.players_tracked > 0);
  const nLast = usable.length > 0 ? usable[usable.length - 1].players_tracked : 0;
  if (nLast <= 0) return [{ n: 1, w: 1 }];
  const ns = usable.map((r) => r.players_tracked);
  const logRange = Math.log(Math.max(...ns) / Math.min(...ns));
  const tLast = Date.parse(usable[usable.length - 1].at);

  if (logRange >= GROWTH_LOG_RANGE) {
    // Identified growth: recency-weighted log-linear trend (recent growth dominates —
    // launch-week inflow decays), projected to the target, centre capped at 300x.
    const t0 = Date.parse(usable[0].at);
    const fit = wols(
      usable.map((r) => (Date.parse(r.at) - t0) / 3_600_000),
      usable.map((r) => Math.log(r.players_tracked)),
      usable.map(
        (r) =>
          Math.exp(-(tLast - Date.parse(r.at)) / (JUMP_RECENCY_HALF_LIFE_HOURS * 3_600_000)),
      ),
    );
    if (fit !== null) {
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
    // Fall through to the refresh branch if the fit fails.
  }

  // Flat history: no trend to fit. Refreshes arrive on the renewal process above;
  // each multiplies the tracked population by a lognormal size whose median scales
  // with staleness: the longer the page sits stale, the bigger the backlog of
  // unsynced players the next refresh picks up.
  //
  // When no jump was ever observed, the last refresh time is unknown (it predates the
  // first read), so the staleness is integrated over one cadence back from the first
  // read — five points, equal weight — instead of trusting the lower bound. A page
  // already stale for hours at the checkpoint is likely stale for longer than the
  // first read shows, which both raises the refresh probability and grows the
  // expected backlog.
  const { stalenessHours, hadJump } = lastRefreshInfo(history, checkpointMs);
  const stalenessGrid = hadJump
    ? [stalenessHours]
    : [0, 1.125, 2.25, 3.375, 4.5].map((d) => stalenessHours + d);
  const meanStaleness =
    stalenessGrid.reduce((a, b) => a + b, 0) / stalenessGrid.length;
  const countDist = new Map<number, number>();
  for (const s of stalenessGrid) {
    for (const { k, w } of refreshCountDistribution(s, horizonHours)) {
      countDist.set(k, (countDist.get(k) ?? 0) + w / stalenessGrid.length);
    }
  }
  const sizeMedian = clamp(1 + meanStaleness, 1.5, 10);
  const sizes = [1.8, 2.5, 3.5, 5, 7, 10];
  const sizeW = sizes.map((s) => lognormalWeight(s, sizeMedian, REFRESH_SIZE_SIGMA));
  const sizeTotal = sizeW.reduce((a, b) => a + b, 0);
  const out: Array<{ n: number; w: number }> = [];
  for (const [k, wk] of [...countDist.entries()].sort((a, b) => a[0] - b[0])) {
    if (k === 0) {
      out.push({ n: nLast, w: wk });
      continue;
    }
    for (let j = 0; j < sizes.length; j++) {
      out.push({
        n: clamp(nLast * Math.pow(sizes[j], k), nLast, nLast * MAX_GROWTH_MULTIPLE),
        w: (wk * sizeW[j]) / sizeTotal,
      });
    }
  }
  const wTotal = out.reduce((a, s) => a + s.w, 0);
  return out.map((s) => ({ ...s, w: s.w / wTotal }));
}

/** One refresh jump for a trophy: the change in earners and players between two
 *  consecutive reads where the tracked population grew. */
interface Jump {
  /** Earners gained across the jump. */
  dE: number;
  /** Players gained across the jump. */
  dN: number;
  /** Earners before the jump. */
  ePrev: number;
  /** Days between the two reads. */
  dtDays: number;
  /** Recency weight (1 at the last read, decaying with a 24 h half-life). */
  recency: number;
}

/** Refresh jumps for one trophy: only intervals where the tracked population grew
 *  identify anything. Flat stretches contribute nothing — the page is frozen between
 *  refreshes, so a flat stretch shows no earning at all. */
function trophyJumps(c: Case, idx: number): Jump[] {
  const jumps: Jump[] = [];
  const hist = c.history;
  const tLast = hist.length > 0 ? Date.parse(hist[hist.length - 1].at) : 0;
  for (let i = 1; i < hist.length; i++) {
    const prev = hist[i - 1];
    const cur = hist[i];
    const rPrev = prev.rates[idx];
    const rCur = cur.rates[idx];
    if (
      rPrev === null ||
      rPrev === undefined ||
      rCur === null ||
      rCur === undefined ||
      prev.players_tracked <= 0 ||
      cur.players_tracked <= prev.players_tracked
    )
      continue;
    const ePrev = (rPrev / 100) * prev.players_tracked;
    const eCur = (rCur / 100) * cur.players_tracked;
    jumps.push({
      dE: eCur - ePrev,
      dN: cur.players_tracked - prev.players_tracked,
      ePrev,
      dtDays: Math.max((Date.parse(cur.at) - Date.parse(prev.at)) / 86_400_000, 1 / 1440),
      recency: Math.exp(
        -(tLast - Date.parse(cur.at)) / (JUMP_RECENCY_HALF_LIFE_HOURS * 3_600_000),
      ),
    });
  }
  return jumps;
}

/** Fit a trophy's marginal joiner rate m and old-cohort earning rate r (per day) from
 *  its refresh jumps, shrunk toward the kind priors. Two passes over
 *  dE = m*dN + r*E_prev*dt: first m (subtracting prior old-cohort earning), then r
 *  (given the fitted m). With no jumps both stay at their priors. */
function estimateMarginalAndEarning(
  jumps: Jump[],
  mPrior: number,
  gPrior: number,
  nLast: number,
): { m: number; r: number } {
  if (jumps.length === 0 || nLast <= 0) return { m: mPrior, r: gPrior };
  const totalJumpN = jumps.reduce((s, j) => s + j.dN, 0);

  // Pass 1: marginal rate. Each jump's m, recency-weighted by jump size.
  let mNum = 0;
  let mDen = 0;
  for (const j of jumps) {
    const mj = clamp((j.dE - gPrior * j.ePrev * j.dtDays) / j.dN, 0, 1);
    const w = j.dN * j.recency;
    mNum += w * mj;
    mDen += w;
  }
  const mRaw = mDen > 0 ? mNum / mDen : mPrior;
  const wFitM = totalJumpN / (totalJumpN + M_PRIOR_STRENGTH_PLAYERS);
  const m = clamp(wFitM * mRaw + (1 - wFitM) * mPrior, 0, 0.99);

  // Pass 2: old-cohort earning rate. Each jump's residual over its gap, weighted by
  // earner-days (intervals with no prior earners carry no information about r).
  let rNum = 0;
  let rDen = 0;
  let totalEarnerDays = 0;
  for (const j of jumps) {
    if (j.ePrev <= 0 || j.dtDays <= 0) continue;
    const rj = clamp((j.dE - m * j.dN) / (j.ePrev * j.dtDays), 0, 2);
    const w = j.ePrev * j.dtDays * j.recency;
    rNum += w * rj;
    rDen += w;
    totalEarnerDays += j.ePrev * j.dtDays;
  }
  if (rDen <= 0) return { m, r: gPrior };
  const rRaw = rNum / rDen;
  const wFitR = totalEarnerDays / (totalEarnerDays + R_PRIOR_STRENGTH_EARNER_DAYS);
  const r = clamp(wFitR * rRaw + (1 - wFitR) * gPrior, 0, 1);
  return { m, r };
}

/** Forecast one trophy's five quantiles (percent, non-decreasing, 4 decimals). */
function forecastTrophy(
  t: Trophy,
  idx: number,
  c: Case,
  scenarios: Array<{ n: number; w: number }>,
  tenureDays: number,
): number[] {
  const horizonDays = c.horizon_minutes / 1440;
  const { kind, count } = classify(t);
  const prior = KIND_PRIOR[kind];

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
    const s = priorMarginal(kind, count, tenureDays, 0);
    const centre = clampRate(100 * s);
    return Z.map((z) => round4(clampRate(invLogitPct(logit(centre) + z * 0.8))));
  }
  const eLast = (lastRate / 100) * lastN;

  // Marginal joiner rate and old-cohort earning rate: fitted from the jumps where
  // the history identifies them, shrunk toward the kind priors where it does not.
  const mPrior = priorMarginal(kind, count, tenureDays, lastRate);
  const { m, r } = estimateMarginalAndEarning(
    trophyJumps(c, idx),
    mPrior,
    prior.g,
    lastN,
  );
  const g = Math.min(1 + r * horizonDays, 2.5);

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
  const checkpointMs = Date.parse(c.at);
  const scenarios = playerScenarios(c.history, horizonHours, horizonDays, checkpointMs);
  const tenureDays = joinerTenureDays(c, horizonDays);
  const trophies = c.trophies.map((t, i) => ({
    id: t.id,
    q: forecastTrophy(t, i, c, scenarios, tenureDays),
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
