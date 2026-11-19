# makefun-gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Tiiashtf/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Tiiashtf/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A deterministic forecaster for GTA VI's PS5 trophy earn rates on Exophase at 2026-11-26T00:00Z. Instead of trending the rate directly, it decomposes each rate into earners / players tracked, fits log-earners and log-players separately against log-time since release (recency-weighted, last 48 h, slopes shrunk toward per-class priors and capped), and takes the ratio at the target. The spread prices fit residual, horizon, trophy class, and — crucially — Exophase refresh risk: a page that has not changed for hours is stale, not stable, so a flat history gets a wide, refresh-aware band (about 10x the house baseline's on the practice cases) instead of the baseline's narrow one. Node 22, standard library only, no network, deterministic, ~0.2 s per case.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=635rx196&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
