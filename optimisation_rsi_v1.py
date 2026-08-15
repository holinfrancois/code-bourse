"""Optimisation des paramètres de la stratégie RSI / EMA(RSI) sur les
cours SRD déjà récupérés (recherche exhaustive / grid search).

Balaie :
- Période du RSI        : 6 à 16 par pas de 2
- Période de l'EMA(RSI)  : 4 à 18 par pas de 2
- Niveau d'entrée du RSI : 30 à 60 par pas de 5

Stratégie testée pour chaque combinaison (identique à backtest_rsi) :
- Achat : le RSI est sous le niveau d'entrée ET repasse au-dessus de sa
  propre EMA.
- Vente : le RSI repasse sous cette EMA.

Pour chaque combinaison, on calcule la somme des rendements (trades
clôturés + rendement latent des positions encore ouvertes) sur
l'ensemble des valeurs du dernier fichier de cours récupéré, puis on
classe les combinaisons par performance pour trouver la meilleure.

⚠️ Avertissement surapprentissage (overfitting) : 336 combinaisons
testées sur seulement ~5 ans d'historique et une trentaine de valeurs,
c'est un terrain propice au surapprentissage — la meilleure combinaison
sur ce passé précis n'a aucune garantie de bien se comporter à l'avenir.
À confirmer sur une autre période ou un autre échantillon de valeurs
avant d'y faire confiance en réel.

Sortie, à côté du script :
- optimisation_rsi_<horodatage>.csv : le classement complet des 336
  combinaisons, du meilleur au pire rendement total combiné.

Pour obtenir le détail des trades et les graphiques de la meilleure
combinaison trouvée, reporte les 3 valeurs affichées dans RSI_PERIOD,
RSI_EMA_PERIOD et ENTRY_RSI_MAX en haut de backtest_rsi_v5.py, puis
relance ce script.
"""

from datetime import datetime
from itertools import product
from pathlib import Path

import pandas as pd

RSI_PERIODS = range(6, 17, 2)
EMA_PERIODS = range(4, 19, 2)
ENTRY_LEVELS = range(30, 61, 5)

OUTPUT_DIR = Path(__file__).resolve().parent


def latest_prices_csv() -> Path:
    candidates = sorted(OUTPUT_DIR.glob("srd_cours_cloture_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            "Aucun fichier srd_cours_cloture_*.csv trouvé dans "
            f"{OUTPUT_DIR}. Lance d'abord le script de récupération des cours."
        )
    return candidates[-1]


def compute_rsi(closes: pd.Series, period: int) -> pd.Series:
    """RSI de Wilder, calculé par lissage exponentiel des gains/pertes."""
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_rsi_ema(rsi: pd.Series, period: int) -> pd.Series:
    """Moyenne mobile exponentielle appliquée au RSI lui-même."""
    return rsi.ewm(span=period, adjust=False).mean()


def find_trade_returns(closes: pd.Series, rsi: pd.Series, rsi_ema: pd.Series, entry_max: int) -> list:
    """Rejoue la stratégie et renvoie le rendement (%) de chaque trade."""
    in_position = False
    entry_price = None
    returns = []

    for dt, price in closes.items():
        r = rsi.get(dt)
        e = rsi_ema.get(dt)
        if pd.isna(r) or pd.isna(e):
            continue
        if not in_position and r < entry_max and r > e:
            in_position = True
            entry_price = price
        elif in_position and r < e:
            returns.append(((price / entry_price - 1) * 100, "clos"))
            in_position = False

    if in_position:
        last_price = closes.iloc[-1]
        returns.append(((last_price / entry_price - 1) * 100, "ouvert"))

    return returns


def main() -> None:
    csv_path = latest_prices_csv()
    print(f"Données utilisées : {csv_path.name}")

    prices = pd.read_csv(csv_path, index_col=0, parse_dates=True, encoding="utf-8-sig")
    tickers = {name: prices[name].dropna() for name in prices.columns}
    print(f"{len(prices)} clôtures x {len(tickers)} valeurs")

    rsi_periods, ema_periods, entry_levels = list(RSI_PERIODS), list(EMA_PERIODS), list(ENTRY_LEVELS)
    n_combos = len(rsi_periods) * len(ema_periods) * len(entry_levels)
    print(
        f"{n_combos} combinaisons à tester "
        f"({len(rsi_periods)} périodes RSI x {len(ema_periods)} périodes EMA "
        f"x {len(entry_levels)} niveaux d'entrée)...\n"
    )

    # RSI et EMA(RSI) mis en cache par (valeur, période) pour ne pas les
    # recalculer à chaque combinaison de niveau d'entrée.
    rsi_cache = {
        rsi_period: {name: compute_rsi(closes, rsi_period) for name, closes in tickers.items()}
        for rsi_period in rsi_periods
    }
    ema_cache = {
        (rsi_period, ema_period): {
            name: compute_rsi_ema(rsi_cache[rsi_period][name], ema_period) for name in tickers
        }
        for rsi_period in rsi_periods
        for ema_period in ema_periods
    }

    results = []
    for rsi_period, ema_period, entry_level in product(rsi_periods, ema_periods, entry_levels):
        closed_sum = latent_sum = 0.0
        n_closed = n_wins = 0

        for name, closes in tickers.items():
            rsi = rsi_cache[rsi_period][name]
            rsi_ema = ema_cache[(rsi_period, ema_period)][name]
            for rendement, statut in find_trade_returns(closes, rsi, rsi_ema, entry_level):
                if statut == "clos":
                    closed_sum += rendement
                    n_closed += 1
                    n_wins += rendement > 0
                else:
                    latent_sum += rendement

        results.append(
            {
                "Période RSI": rsi_period,
                "Période EMA(RSI)": ema_period,
                "Niveau entrée RSI": entry_level,
                "Nb trades clôturés": n_closed,
                "Somme rendements clôturés (%)": round(closed_sum, 2),
                "Somme rendements latents (%)": round(latent_sum, 2),
                "Total combiné (%)": round(closed_sum + latent_sum, 2),
                "Taux de réussite (%)": round(n_wins / n_closed * 100, 1) if n_closed else None,
            }
        )

    grid = pd.DataFrame(results).sort_values("Total combiné (%)", ascending=False).reset_index(drop=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_csv = OUTPUT_DIR / f"optimisation_rsi_{timestamp}.csv"
    grid.to_csv(grid_csv, index=False, encoding="utf-8-sig")

    print("Top 15 des combinaisons (classées par rendement total combiné) :\n")
    print(grid.head(15).to_string(index=False))

    best = grid.iloc[0]
    print(
        "\nMeilleure configuration trouvée :\n"
        f"- Période RSI              : {int(best['Période RSI'])}\n"
        f"- Période EMA(RSI)          : {int(best['Période EMA(RSI)'])}\n"
        f"- Niveau d'entrée RSI       : {int(best['Niveau entrée RSI'])}\n"
        f"- Trades clôturés           : {int(best['Nb trades clôturés'])} "
        f"(taux de réussite {best['Taux de réussite (%)']}%)\n"
        f"- Rendement total combiné (toutes valeurs, clôturé + latent) : "
        f"{best['Total combiné (%)']:+.2f} %"
    )
    print(f"\nClassement complet ({len(grid)} combinaisons) : {grid_csv.name}")
    print(
        "\n⚠️ Avertissement : cette combinaison est la meilleure SUR CE PASSÉ "
        "précis (mêmes 5 ans, mêmes valeurs qui ont servi à la choisir). Avec "
        f"{n_combos} combinaisons testées, le risque de surapprentissage est "
        "réel — à valider sur une autre période ou un autre échantillon de "
        "valeurs avant d'y faire confiance en réel."
    )


if __name__ == "__main__":
    main()
