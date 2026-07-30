"""
generate_dataset.py
--------------------
Generates a synthetic but realistic League of Legends ranked match dataset,
modeled on the structure of the well-known "High Diamond Ranked 10 Min" LoL
dataset (blue/red team early-game stats -> match outcome).

Each row = one ranked match, viewed from the Blue team's perspective at the
10-minute mark. Features are correlated with the outcome (blueWins) the way
real LoL data behaves (e.g. more gold/kills/towers => higher win probability),
with realistic noise added so the classification task is non-trivial
(~72-76% achievable test accuracy with logistic regression, matching the
real dataset's difficulty).

Run:
    python generate_dataset.py
Produces:
    league_of_legends_data_stats.csv
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
N_SAMPLES = 9879  # matches the size of the original Kaggle dataset

np.random.seed(RNG_SEED)


def generate_lol_dataset(n=N_SAMPLES, seed=RNG_SEED):
    rng = np.random.default_rng(seed)

    # ---- Latent "skill gap" driving the match (not directly observable) ----
    skill_gap = rng.normal(0, 1, n)  # positive => blue team advantaged

    def noisy(base, scale, minv=None):
        val = base + rng.normal(0, scale, n)
        if minv is not None:
            val = np.clip(val, minv, None)
        return val

    # ---- Blue team early-game stats (10 min) ----
    blueWardsPlaced   = noisy(20 + 2 * skill_gap, 5, 0).round().astype(int)
    blueWardsDestroyed= noisy(2 + 0.5 * skill_gap, 1.5, 0).round().astype(int)
    blueFirstBlood    = (rng.random(n) < (0.5 + 0.12 * skill_gap)).astype(int)
    blueKills         = noisy(6 + 2.2 * skill_gap, 2.2, 0).round().astype(int)
    blueDeaths        = noisy(6 - 2.2 * skill_gap, 2.2, 0).round().astype(int)
    blueAssists       = noisy(6 + 2.0 * skill_gap, 2.5, 0).round().astype(int)
    blueEliteMonsters = noisy(0.5 + 0.3 * skill_gap, 0.6, 0).round().astype(int)
    blueDragons       = noisy(0.3 + 0.2 * skill_gap, 0.5, 0).round().astype(int)
    blueHeralds       = noisy(0.2 + 0.1 * skill_gap, 0.4, 0).round().astype(int)
    blueTowersDestroyed = noisy(0.5 + 0.3 * skill_gap, 0.7, 0).round().astype(int)
    blueTotalGold     = noisy(16500 + 1400 * skill_gap, 1200, 0).round().astype(int)
    blueAvgLevel      = noisy(6.9 + 0.25 * skill_gap, 0.35, 1)
    blueTotalExperience = noisy(17500 + 1300 * skill_gap, 1300, 0).round().astype(int)
    blueTotalMinionsKilled = noisy(220 + 12 * skill_gap, 20, 0).round().astype(int)
    blueTotalJungleMinionsKilled = noisy(55 + 6 * skill_gap, 10, 0).round().astype(int)
    blueGoldDiff      = noisy(0 + 1800 * skill_gap, 900)
    blueExperienceDiff= noisy(0 + 1500 * skill_gap, 900)
    blueCSPerMin      = blueTotalMinionsKilled / 10.0
    blueGoldPerMin    = blueTotalGold / 10.0

    # ---- Red team early-game stats (roughly mirror-image) ----
    redWardsPlaced    = noisy(20 - 2 * skill_gap, 5, 0).round().astype(int)
    redWardsDestroyed = noisy(2 - 0.5 * skill_gap, 1.5, 0).round().astype(int)
    redFirstBlood     = 1 - blueFirstBlood
    redKills          = blueDeaths.copy()
    redDeaths         = blueKills.copy()
    redAssists        = noisy(6 - 2.0 * skill_gap, 2.5, 0).round().astype(int)
    redEliteMonsters  = noisy(0.5 - 0.3 * skill_gap, 0.6, 0).round().astype(int)
    redDragons        = noisy(0.3 - 0.2 * skill_gap, 0.5, 0).round().astype(int)
    redHeralds        = noisy(0.2 - 0.1 * skill_gap, 0.4, 0).round().astype(int)
    redTowersDestroyed= noisy(0.5 - 0.3 * skill_gap, 0.7, 0).round().astype(int)
    redTotalGold      = noisy(16500 - 1400 * skill_gap, 1200, 0).round().astype(int)
    redAvgLevel       = noisy(6.9 - 0.25 * skill_gap, 0.35, 1)
    redTotalExperience= noisy(17500 - 1300 * skill_gap, 1300, 0).round().astype(int)
    redTotalMinionsKilled = noisy(220 - 12 * skill_gap, 20, 0).round().astype(int)
    redTotalJungleMinionsKilled = noisy(55 - 6 * skill_gap, 10, 0).round().astype(int)
    redCSPerMin       = redTotalMinionsKilled / 10.0
    redGoldPerMin     = redTotalGold / 10.0

    # ---- Outcome: logistic function of the underlying skill gap + key stats ----
    logit = (
        1.15 * skill_gap
        + 0.00035 * (blueTotalGold - redTotalGold)
        + 0.09 * (blueKills - blueDeaths)
        + 0.35 * (blueDragons - redDragons)
        + 0.25 * (blueHeralds - redHeralds)
        + 0.20 * (blueTowersDestroyed - redTowersDestroyed)
        + rng.normal(0, 0.9, n)  # irreducible noise -> realistic ~73% ceiling
    )
    prob_blue_win = 1 / (1 + np.exp(-logit))
    blueWins = (rng.random(n) < prob_blue_win).astype(int)

    df = pd.DataFrame({
        "blueWins": blueWins,
        "blueWardsPlaced": blueWardsPlaced,
        "blueWardsDestroyed": blueWardsDestroyed,
        "blueFirstBlood": blueFirstBlood,
        "blueKills": blueKills,
        "blueDeaths": blueDeaths,
        "blueAssists": blueAssists,
        "blueEliteMonsters": blueEliteMonsters,
        "blueDragons": blueDragons,
        "blueHeralds": blueHeralds,
        "blueTowersDestroyed": blueTowersDestroyed,
        "blueTotalGold": blueTotalGold,
        "blueAvgLevel": blueAvgLevel.round(2),
        "blueTotalExperience": blueTotalExperience,
        "blueTotalMinionsKilled": blueTotalMinionsKilled,
        "blueTotalJungleMinionsKilled": blueTotalJungleMinionsKilled,
        "blueGoldDiff": blueGoldDiff.round(0).astype(int),
        "blueExperienceDiff": blueExperienceDiff.round(0).astype(int),
        "blueCSPerMin": blueCSPerMin.round(2),
        "blueGoldPerMin": blueGoldPerMin.round(2),
        "redWardsPlaced": redWardsPlaced,
        "redWardsDestroyed": redWardsDestroyed,
        "redFirstBlood": redFirstBlood,
        "redKills": redKills,
        "redDeaths": redDeaths,
        "redAssists": redAssists,
        "redEliteMonsters": redEliteMonsters,
        "redDragons": redDragons,
        "redHeralds": redHeralds,
        "redTowersDestroyed": redTowersDestroyed,
        "redTotalGold": redTotalGold,
        "redAvgLevel": redAvgLevel.round(2),
        "redTotalExperience": redTotalExperience,
        "redTotalMinionsKilled": redTotalMinionsKilled,
        "redTotalJungleMinionsKilled": redTotalJungleMinionsKilled,
        "redCSPerMin": redCSPerMin.round(2),
        "redGoldPerMin": redGoldPerMin.round(2),
    })

    return df


if __name__ == "__main__":
    df = generate_lol_dataset()
    out_path = "league_of_legends_data_stats.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows x {len(df.columns)} columns to {out_path}")
    print(df["blueWins"].value_counts(normalize=True))
