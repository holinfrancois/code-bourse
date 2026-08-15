"""Backtest simple d'une stratégie RSI sur les cours SRD déjà récupérés.

Stratégie long-only, une position à la fois par valeur :
- Entrée : le RSI hebdomadaire passe sous le seuil de survente (30).
- Sortie : le RSI hebdomadaire passe au-dessus du seuil de surachat (70).

Ce script réutilise le dernier export CSV produit par fetch_srd_prices.py
(fichier srd_cours_cloture_*.csv le plus récent du dossier) plutôt que de
retélécharger les données.

Avertissement : avec ~1 an de données hebdomadaires (une cinquantaine de
points par valeur), le nombre de signaux générés est très faible et les
résultats ci-dessous sont indicatifs, pas statistiquement robustes. Un
backtest fiable demandera plusieurs années d'historique.
"""

from pathlib import Path

import pandas as pd

RSI_PERIOD = 14
OVERSOLD = 30
OVERBOUGHT = 70

OUTPUT_DIR = Path(__file__).resolve().parent


def latest_prices_csv() -> Path:
    candidates = sorted(OUTPUT_DIR.glob("srd_cours_cloture_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            "Aucun fichier srd_cours_cloture_*.csv trouvé dans "
            f"{OUTPUT_DIR}. Lance d'abord fetch_srd_prices.py."
        )
    return candidates[-1]


def compute_rsi(closes: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """RSI de Wilder, calculé par lissage exponentiel des gains/pertes."""
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def backtest_ticker(closes: pd.Series) -> dict:
    """Simule la stratégie sur une série de clôtures et résume les trades."""
    closes = closes.dropna()
    rsi = compute_rsi(closes)

    in_position = False
    entry_price = None
    trades = []

    for dt, price in closes.items():
        r = rsi.get(dt)
        if pd.isna(r):
            continue
        if not in_position and r < OVERSOLD:
            in_position = True
            entry_price = price
        elif in_position and r > OVERBOUGHT:
            trades.append((price / entry_price) - 1)
            in_position = False
            entry_price = None

    total_return = 1.0
    for t in trades:
        total_return *= 1 + t
    total_return -= 1

    wins = sum(1 for t in trades if t > 0)

    return {
        "Nb trades": len(trades),
        "Position ouverte fin période": in_position,
        "Rendement total (%)": round(total_return * 100, 2) if trades else 0.0,
        "Rendement moyen/trade (%)": round(sum(trades) / len(trades) * 100, 2) if trades else None,
        "Taux de réussite (%)": round(wins / len(trades) * 100, 1) if trades else None,
    }


def main() -> None:
    csv_path = latest_prices_csv()
    print(f"Données utilisées : {csv_path.name}")

    prices = pd.read_csv(csv_path, index_col=0, parse_dates=True, encoding="utf-8-sig")
    print(f"{len(prices)} clôtures x {len(prices.columns)} valeurs\n")

    results = {name: backtest_ticker(prices[name]) for name in prices.columns}
    summary = pd.DataFrame(results).T.sort_values("Rendement total (%)", ascending=False)

    print(f"Backtest RSI (achat RSI < {OVERSOLD}, vente RSI > {OVERBOUGHT}) :\n")
    print(summary)

    traded = summary[summary["Nb trades"] > 0]
    print(f"\n{len(traded)}/{len(summary)} valeurs ont généré au moins un trade sur la période.")
    print(
        "Avertissement : historique d'à peine 1 an en hebdomadaire, trop "
        "court pour des conclusions fiables — à relancer sur un historique "
        "plus long avant d'accorder du poids à ces chiffres."
    )


if __name__ == "__main__":
    main()
