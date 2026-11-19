// gta6-trophy-rarity-forecast-01 — earners/players decomposition forecaster (Node 22, stdlib only).
//
// Forecast, for each of the game's 55 PS5 trophies, five quantiles of its Exophase earn rate
// (percent, 0-100) at the case target. Score is mean pinball loss on the logit scale.
//
// MODEL — rate = earners / players_tracked, so forecast the two counts separately:
//   * Earners E(t): log E is fit (weighted least squares, recency-weighted, last-48h window)
//     against log-time x = ln(1 + hours since release 2026-11-19T00:00Z). Log-time grows
//     sublinearly in wall time, so early explosive growth damps as the launch cohort ages.
//   * Players N(t): log N is fit the same way. N never shrinks; its fitted slope is capped.
//   * Median rate = 100 * E_hat / N_hat. This beats forecasting the rate directly because
//     the two counts move for different reasons: N jumps when Exophase's tracked-player pool
//     grows (sales + PSN syncs), E grows as the tracked cohort plays through the game. On
//     launch night the house saw exactly this: N 638 -> 3868 at the first refresh while
//     earners crept, so every rate but the 100% one fell.
//   * Trophy class (from description + tier) only sets priors/caps, never replaces the fit:
//     story = chapter/final-mission completions (fast early E growth, then saturation),
//     activity = one-off feats (medium), grind = 250 stunts / gold medals / 100% completion /
//     collectibles (slow), platinum = slowest of all (needs everything).
//   * Fitted slopes are shrunk toward the class prior when the window has few distinct reads,
//     and capped per class, so a steep launch-night slope cannot explode over a 4.5-day horizon.
//
// SPREAD — pinball loss is a proper scoring rule, so the quantiles must be the forecast the
// author really believes. The logit-space spread s combines:
//   * s_fit: the weighted residual spread of the log-E and log-N fits (a fit that wobbles
//     earns a wide band), floored so a perfect-looking fit is never trusted completely;
//   * s_horizon = 0.30 * sqrt(days): trend uncertainty grows with the horizon;
//   * s_refresh: Exophase refreshes counts on its own schedule, not with every player's
//     trophy — a page that has not changed for hours is STALE, not stable. The practice
//     cases prove it: a flat page (638 players, identical rates for 3+ hours) was followed
//     by a refresh that moved every rate by ~1.8 logit units (N x6.06 in 4.7h). A degenerate
//     (flat) history therefore gets a wide band (s_refresh = 1.25) and a stale-N adjustment
//     to the median (below); a rich history gets only a small term for the time since the
//     last observed change.
//   * s_class: a little extra for grind/platinum trophies, whose earners arrive in lumps.
// On a flat practice history the 90% band is about +/-2 logit units wide, versus the house
// baseline's +/-0.2 — the baseline's coverage on practice was ~0 because the refresh landed
// outside its band; this band is built to contain it.
//
// STALE-PAGE MEDIAN (degenerate histories only): when every read is identical, the displayed
// N is hours old. The true N at the checkpoint is already larger, and the next refresh
// exposes it. So N_hat is grown at the launch-week prior (doubling every 4h, capped at 6x —
// the observed first refresh was 6.06x in 4.7h) over the stale period plus the horizon, and
// E_hat is grown a little by class. On a rich history this never triggers: the fit rules.
//
// CALIBRATION (live Exophase page, read 2026-11-19 ~16:49 UTC, ~17h after release):
//   platinum 0.63% -> 0.03%, Chapter 1 1.25% -> 0.73%, Chapter 2 1.1% -> 0.14%,
//   Chapters 3/4 and the final mission 1.1% -> 0.06%, most other trophies 0.63-2.04% ->
//   0.03-0.24%, while Gone Fishing stayed ~100% (99.99%) and Beat Cal 2.04% -> 1.89%.
//   So by day 1 the denominator had exploded while earners grew slowly, story progress
//   drops steeply by chapter, and grind trophies sit at the ~0.03% floor. That is why the
//   caps above are tight for grind/platinum and why the median is a ratio, not a trend.
//
// Deterministic: no randomness, no Date.now() in the forecast (only the case's own
// timestamps), no environment dependence beyond the input/output dirs. Fast: O(reads x
// trophies), far under the 8 s cap. Robust: null rates are skipped per trophy, a missing or
// zero players_tracked falls back to the last known N, a single-read or empty history falls
// back to persistence with a wide band, and any unexpected input still yields a valid
// forecast (per-case try/catch -> persistence fallback).

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Release instant: the case's game.release_date at 00:00 UTC (fallback: 2026-11-19). */
const FALLBACK_RELEASE_MS = Date.parse('2026-11-19T00:00:00Z');

type TrophyClass = 'story' | 'activity' | 'grind' | 'platinum';

interface TrophyInfo {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}
interface Read {
  at: string;
  players_tracked: number | null;
  trophies_earned?: number | null;
  rates: Array<number | null>;
}
interface Case {
  id?: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { name: string; platform: string; release_date: string; source: string };
  trophies: TrophyInfo[];
  history: Read[];
  notes?: string;
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const sigmoidPct = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals, half-up like the starter. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));

/** Inverse standard normal CDF (Acklam's approximation); used only if a case asks for
 *  levels other than the contest's five, so the standard levels keep the exact Z above. */
function normInv(p: number): number {
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239e0];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0, -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0, 3.754408661907416e0];
  const plow = 0.02425;
  const phigh = 1 - plow;
  let q: number;
  let r: number;
  if (p < plow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > phigh) {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  q = p - 0.5;
  r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

/** Classify a trophy from its description and tier. Priors/caps only; the fit rules. */
function classify(t: TrophyInfo): TrophyClass {
  if (t.tier === 'platinum') return 'platinum';
  const d = (t.description ?? '').toLowerCase();
  if (/chapter|final mission|complete the final mission for/.test(d)) return 'story';
  if (/100% completion|250 stunts|gold medal|collectible card|meditate|hunt a unique|waink it at 100|discover 100|purchase all|earn \$|spend \$|complete 50 challenges|complete 8 mission strands/.test(d)) return 'grind';
  if (t.tier === 'gold') return 'grind'; // e.g. 100% completion
  return 'activity';
}

/** Per-class prior slope of log E against log-time, max fitted slope, and per-hour E growth
 *  used only for the stale-page fallback (degenerate histories). Story trophies grow fast in
 *  the first hours — new players finish Chapter 1 within a few hours of starting — while
 *  grind trophies barely move (nobody does 250 stunts or 100% completion on launch night);
 *  the live page at ~17h after release confirms the split: Chapter 1 held at 0.73% while
 *  grind trophies sat at the 0.03% floor. */
const CLASS: Record<TrophyClass, { priorSlope: number; maxSlope: number; stalePerHour: number; staleCap: number }> = {
  story: { priorSlope: 1.2, maxSlope: 3.5, stalePerHour: 0.35, staleCap: 4 },
  activity: { priorSlope: 0.7, maxSlope: 2.0, stalePerHour: 0.12, staleCap: 2 },
  grind: { priorSlope: 0.3, maxSlope: 1.0, stalePerHour: 0.03, staleCap: 1.3 },
  platinum: { priorSlope: 0.15, maxSlope: 0.6, stalePerHour: 0, staleCap: 1 },
};
/** Prior slope / cap for log N against log-time (class-independent). */
const N_PRIOR_SLOPE = 0.8;
const N_MAX_SLOPE = 2.0;
/** Recency weighting: half-life 12 h; window: last 48 h of reads. */
const HALF_LIFE_H = 12;
const WINDOW_H = 48;

interface Fit { slope: number; intercept: number; resid: number; n: number; distinct: number }

/** Weighted least squares of y on x. Returns null when too few distinct x values. */
function wls(xs: number[], ys: number[], ws: number[]): Fit | null {
  const distinct = new Set(xs.map((x) => x.toFixed(6))).size;
  if (xs.length < 3 || distinct < 3) return null;
  let sw = 0;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < xs.length; i++) { sw += ws[i]; sx += ws[i] * xs[i]; sy += ws[i] * ys[i]; }
  const mx = sx / sw;
  const my = sy / sw;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < xs.length; i++) {
    sxx += ws[i] * (xs[i] - mx) * (xs[i] - mx);
    sxy += ws[i] * (xs[i] - mx) * (ys[i] - my);
  }
  if (sxx <= 1e-12) return null;
  const slope = sxy / sxx;
  const intercept = my - slope * mx;
  let se = 0;
  for (let i = 0; i < xs.length; i++) {
    const r = ys[i] - (intercept + slope * xs[i]);
    se += ws[i] * r * r;
  }
  const resid = Math.sqrt(se / sw);
  return { slope, intercept, resid, n: xs.length, distinct };
}

/** Shrink a fitted slope toward its class prior: the fewer distinct reads, the more prior.
 *  Then clamp to [0, maxSlope] (counts never shrink, growth never explodes). */
function shrunkSlope(fit: Fit | null, prior: number, maxSlope: number): { slope: number; intercept: number; resid: number } {
  if (fit === null) return { slope: prior, intercept: 0, resid: 0 };
  const w = clamp(fit.distinct / 8, 0, 1);
  const slope = clamp(w * fit.slope + (1 - w) * prior, 0, maxSlope);
  return { slope, intercept: fit.intercept, resid: fit.resid };
}

/** Forecast one trophy. Returns [medianRate, spread] in percent / logit units. */
function forecastTrophy(
  idx: number,
  trophy: TrophyInfo,
  c: Case,
  releaseMs: number,
  targetMs: number,
  checkpointMs: number,
): [number, number] {
  const cls = classify(trophy);
  const cp = CLASS[cls];
  const days = Math.max(0, c.horizon_minutes / 1440);
  const horizonH = Math.max(0, c.horizon_minutes / 60);

  // Collect the trophy's series: (log-time x, N, E). Null rates are skipped for this trophy;
  // a missing/zero players_tracked falls back to the last known N (a read without N still
  // carries rates, and E needs an N).
  const xs: number[] = [];
  const ns: number[] = [];
  const es: number[] = [];
  const times: number[] = [];
  let lastN = 0;
  let lastRate: number | null = null;
  let firstReadMs = checkpointMs;
  let lastChangeMs = checkpointMs;
  let prevN: number | null = null;
  let prevRate: number | null = null;
  const history = Array.isArray(c.history) ? c.history : [];
  for (const h of history) {
    if (!h || typeof h.at !== 'string') continue;
    const t = Date.parse(h.at);
    if (!Number.isFinite(t)) continue;
    if (times.length === 0) firstReadMs = t;
    const n = typeof h.players_tracked === 'number' && h.players_tracked > 0 ? h.players_tracked : lastN;
    if (typeof h.players_tracked === 'number' && h.players_tracked > 0) lastN = h.players_tracked;
    const rate = Array.isArray(h.rates) ? h.rates[idx] : null;
    if (typeof rate === 'number' && Number.isFinite(rate)) {
      lastRate = rate;
      if (prevRate !== null && rate !== prevRate) lastChangeMs = t;
      prevRate = rate;
    }
    if (typeof h.players_tracked === 'number' && h.players_tracked > 0) {
      if (prevN !== null && h.players_tracked !== prevN) lastChangeMs = t;
      prevN = h.players_tracked;
    }
    if (typeof rate === 'number' && Number.isFinite(rate) && n > 0) {
      const x = Math.log(1 + Math.max(0, (t - releaseMs) / 3_600_000));
      xs.push(x);
      ns.push(n);
      es.push((rate / 100) * n);
      times.push(t);
    }
  }
  const distinctN = new Set(ns).size;
  const distinctPairs = new Set(es.map((e, i) => `${e.toFixed(4)}|${ns[i]}`)).size;
  const degenerate = xs.length < 3 || distinctN <= 2 || distinctPairs <= 2;

  let median: number;
  let sFit = 0;
  let sRefresh: number;

  if (degenerate) {
    // Stale page: every read identical, so the displayed N is hours old. Grow N at the
    // launch-week prior (doubling every 3 h, capped at 6x — the observed first refresh was
    // 638 -> 3868, 6.06x in 4.7 h, i.e. doubling about every 1.8 h; 3 h is the conservative
    // side of that) over the stale period plus the horizon; grow E by class (story trophies
    // are finished by new players within hours, grind trophies barely move). The practice
    // answers are the refreshed page, so the median must sit near the refreshed level, not
    // near the stale displayed rate — but the wide band below still covers the case where
    // the refresh has not landed yet. Median = last rate scaled.
    const staleH = Math.max(0, (checkpointMs - firstReadMs) / 3_600_000);
    const nFactor = clamp(Math.pow(2, (staleH + horizonH) / 3), 1, 6);
    const eFactor = clamp(1 + cp.stalePerHour * (staleH + horizonH), 1, cp.staleCap);
    const base = lastRate !== null ? lastRate : (cls === 'platinum' ? 0.03 : cls === 'grind' ? 0.1 : 0.5);
    median = base * (eFactor / nFactor);
    // The 100% trophy: earners are the whole pool, so it stays ~100% (N growth cannot
    // dilute a trophy everyone has); only a hair below 100 for the few who never do it.
    if (lastRate !== null && lastRate >= 99.99) median = 99.95;
    sRefresh = 1.25;
  } else {
    // Rich history: fit log E and log N against log-time over the last 48 h, recency-weighted.
    const cutoff = checkpointMs - WINDOW_H * 3_600_000;
    const wx: number[] = [];
    const wyE: number[] = [];
    const wyN: number[] = [];
    const ww: number[] = [];
    for (let i = 0; i < xs.length; i++) {
      if (times[i] < cutoff && xs.length > 6) continue;
      const w = Math.pow(0.5, (checkpointMs - times[i]) / (HALF_LIFE_H * 3_600_000));
      wx.push(xs[i]);
      wyE.push(Math.log(es[i] + 0.5)); // +0.5 pseudo-count so a zero-earner read still fits
      wyN.push(Math.log(ns[i]));
      ww.push(w);
    }
    const fitE = wx.length >= 3 ? wls(wx, wyE, ww) : null;
    const fitN = wx.length >= 3 ? wls(wx, wyN, ww) : null;
    if (wx.length < 3 || fitE === null || fitN === null) {
      // Too little usable window to fit: fall back to persistence with a wide band rather
      // than extrapolating from an unanchored line.
      median = lastRate !== null ? lastRate : 0.5;
      if (lastRate !== null && lastRate >= 99.99) median = 99.95;
      sFit = 0;
      sRefresh = 1.25;
    } else {
      const eFit = shrunkSlope(fitE, cp.priorSlope, cp.maxSlope);
      const nFit = shrunkSlope(fitN, N_PRIOR_SLOPE, N_MAX_SLOPE);
      // Re-anchor each intercept at the shrunk slope (weighted mean of y - slope*x), so the
      // level follows the recent reads rather than the unshrunk line.
      const anchor = (ys: number[], slope: number): number => {
        let sw = 0;
        let s = 0;
        for (let i = 0; i < wx.length; i++) { sw += ww[i]; s += ww[i] * (ys[i] - slope * wx[i]); }
        return sw > 0 ? s / sw : 0;
      };
      const xTarget = Math.log(1 + Math.max(0, (targetMs - releaseMs) / 3_600_000));
      const eHat = Math.exp(anchor(wyE, eFit.slope) + eFit.slope * xTarget);
      let nHat = Math.exp(anchor(wyN, nFit.slope) + nFit.slope * xTarget);
      const nLast = ns[ns.length - 1];
      nHat = clamp(nHat, nLast, nLast * Math.exp(N_MAX_SLOPE * Math.max(0.01, xTarget - wx[wx.length - 1])));
      const eLast = es[es.length - 1];
      const eCap = Math.max(eLast * Math.exp(cp.maxSlope * Math.max(0.01, xTarget - wx[wx.length - 1])), eLast + 0.5);
      const eClamped = clamp(eHat, 0, Math.min(eCap, 0.999 * nHat));
      median = (eClamped / nHat) * 100;
      if (lastRate !== null && lastRate >= 99.99 && median < 99) median = Math.max(median, 99.5);
      sFit = Math.sqrt((fitE ? fitE.resid : 0.3) ** 2 + (fitN ? fitN.resid : 0.3) ** 2);
      // Refresh risk on a rich history: small, proportional to how long the page has sat
      // unchanged relative to a ~12 h refresh cadence.
      const sinceChangeH = Math.max(0, (checkpointMs - lastChangeMs) / 3_600_000);
      sRefresh = 0.35 * clamp(sinceChangeH / 12, 0, 1);
    }
  }

  const sClass = cls === 'grind' || cls === 'platinum' ? 0.15 : 0;
  const s = clamp(Math.sqrt(Math.max(0.15, sFit) ** 2 + (0.3 * Math.sqrt(days)) ** 2 + sRefresh ** 2 + sClass ** 2), 0.35, 2.0);
  return [clamp(median, 0.01, 99.99), s];
}

/** Fallback forecast for a case that cannot be parsed: persistence with a wide band. */
function fallbackRows(c: Case): Array<{ id: number; q: number[] }> {
  const z = c.levels && c.levels.length === 5 ? Z : c.levels.map((l) => normInv(l));
  const last = Array.isArray(c.history) && c.history.length > 0 ? c.history[c.history.length - 1] : null;
  return c.trophies.map((t, i) => {
    const rate = last && Array.isArray(last.rates) && typeof last.rates[i] === 'number' ? last.rates[i] : 0.5;
    const l = logit(rate);
    const q = z.map((zv) => round4(clamp(sigmoidPct(l + zv * 1.25), 0, 100)));
    for (let j = 1; j < q.length; j++) if (q[j] < q[j - 1]) q[j] = q[j - 1];
    return { id: t.id, q };
  });
}

function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  const levels = Array.isArray(c.levels) && c.levels.length === 5 ? c.levels : LEVELS;
  const z = levels.length === LEVELS.length && levels.every((t, j) => t === LEVELS[j]) ? Z : levels.map((l) => normInv(clamp(l, 0.0001, 0.9999)));
  const releaseMs = c.game && c.game.release_date ? Date.parse(`${c.game.release_date}T00:00:00Z`) : FALLBACK_RELEASE_MS;
  const rel = Number.isFinite(releaseMs) ? releaseMs : FALLBACK_RELEASE_MS;
  const targetMs = Date.parse(c.target);
  const checkpointMs = Date.parse(c.at);
  const tgt = Number.isFinite(targetMs) ? targetMs : checkpointMs + (c.horizon_minutes || 0) * 60_000;
  const chk = Number.isFinite(checkpointMs) ? checkpointMs : rel;
  return c.trophies.map((t, i) => {
    const [median, s] = forecastTrophy(i, t, c, rel, tgt, chk);
    const l = logit(median);
    const q = z.map((zv) => round4(clamp(sigmoidPct(l + zv * s), 0, 100)));
    // Non-decreasing guard (rounding can tie adjacent quantiles; never let one fall below).
    for (let j = 1; j < q.length; j++) if (q[j] < q[j - 1]) q[j] = q[j - 1];
    return { id: t.id, q };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  let trophies: Array<{ id: number; q: number[] }>;
  try {
    const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
    if (!Array.isArray(c.trophies) || c.trophies.length === 0) throw new Error(`${file} has no trophies`);
    trophies = forecastCase(c);
  } catch (err) {
    // Never crash: an unreadable case still gets a valid forecast shape from persistence.
    try {
      const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
      trophies = fallbackRows(c);
    } catch {
      trophies = [];
    }
    if (trophies.length === 0) {
      // No trophies can be recovered at all; write an empty valid object rather than failing.
      writeFileSync(join(OUTPUT_DIR, `${id}.out`), '{"trophies":[]}\n');
      continue;
    }
    void err;
  }
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
