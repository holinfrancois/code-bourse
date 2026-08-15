"""Optimisation d'une stratégie de croisement de deux moyennes mobiles du
RSI (façon MACD appliqué au RSI plutôt qu'au prix), sur les cours SRD
déjà récupérés. Recherche exhaustive / grid search, puis génération
automatique du détail des trades et des graphiques pour la meilleure
combinaison trouvée.

Stratégie testée, long-only, une position à la fois par valeur :
- Achat : la moyenne mobile exponentielle COURTE du RSI passe au-dessus
  de la moyenne mobile exponentielle LONGUE du RSI.
- Vente : la moyenne courte repasse sous la moyenne longue.

Balaie :
- Période du RSI                         : 6 à 16 par pas de 2
- Périodes des deux EMA(RSI), courte/longue : 4 à 18 par pas de 2,
  toutes les paires (courte < longue)
(168 combinaisons)

Pour chaque combinaison, on calcule la somme des rendements (trades
clôturés + rendement latent des positions encore ouvertes) sur
l'ensemble des valeurs du dernier fichier de cours récupéré, puis on
classe les combinaisons par performance.

La meilleure combinaison est ensuite rejouée en détail : trades,
résumé avec ligne TOTAL, et un graphique par valeur — les graphiques
affichent en titre les paramètres retenus (période RSI, périodes des
deux EMA).

⚠️ Avertissement surapprentissage (overfitting) : 168 combinaisons
testées sur seulement ~5 ans d'historique et une trentaine de valeurs,
c'est un terrain propice au surapprentissage — la meilleure combinaison
sur ce passé précis n'a aucune garantie de bien se comporter à l'avenir.

Sorties, à côté du script :
- optimisation_macd_rsi_<horodatage>.csv : le classement complet des
  168 combinaisons.
- trades_macd_rsi_meilleure_config_<horodatage>.csv/.xlsx : détail des
  trades et résumé (avec TOTAL) pour la meilleure combinaison.
- graphiques_macd_rsi_meilleure_config_<horodatage>/ : un PNG par
  valeur pour la meilleure combinaison, paramètres affichés en titre.
"""

from datetime import datetime
from itertools import combinations, product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RSI_PERIODS = range(6, 17, 2)
EMA_PERIODS = range(4, 19, 2)

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


def find_trades(name: str, closes: pd.Series, ema_short: pd.Series, ema_long: pd.Series) -> list:
    """Rejoue la stratégie et renvoie le détail de chaque trade.

    Modèle à une position à la fois : sortir signifie que la courte est
    forcément repassée sous la longue, donc la ré-entrée suivante exige
    mécaniquement un nouveau croisement à la hausse — pas besoin de
    comparer explicitement au point précédent.
    """
    in_position = False
    entry_date = entry_price = None
    trades = []

    for dt, price in closes.items():
        es = ema_short.get(dt)
        el = ema_long.get(dt)
        if pd.isna(es) or pd.isna(el):
            continue
        if not in_position and es > el:
            in_position = True
            entry_date, entry_price = dt, price
        elif in_position and es < el:
            trades.append(
                {
                    "Valeur": name,
                    "Date entrée": entry_date,
                    "Prix entrée": round(entry_price, 2),
                    "Date sortie": dt,
                    "Prix sortie": round(price, 2),
                    "Rendement (%)": round((price / entry_price - 1) * 100, 2),
                    "Semaines détenues": max((dt - entry_date).days // 7, 1),
                    "Statut": "Clôturé",
                }
            )
            in_position = False

    if in_position:
        last_date, last_price = closes.index[-1], closes.iloc[-1]
        trades.append(
            {
                "Valeur": name,
                "Date entrée": entry_date,
                "Prix entrée": round(entry_price, 2),
                "Date sortie": None,
                "Prix sortie": None,
                "Rendement (%)": round((last_price / entry_price - 1) * 100, 2),
                "Semaines détenues": max((last_date - entry_date).days // 7, 1),
                "Statut": "Ouvert (non clôturé, au dernier cours connu)",
            }
        )

    return trades


def plot_ticker(
    name: str,
    closes: pd.Series,
    rsi: pd.Series,
    ema_short: pd.Series,
    ema_long: pd.Series,
    trades: list,
    charts_dir: Path,
    rsi_period: int,
    short_period: int,
    long_period: int,
) -> Path:
    fig, (ax_price, ax_rsi) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_price.plot(closes.index, closes.values, color="#1f77b4", linewidth=1.2, label="Clôture")

    entries = [(t["Date entrée"], t["Prix entrée"]) for t in trades]
    exits = [(t["Date sortie"], t["Prix sortie"]) for t in trades if t["Date sortie"] is not None]

    if entries:
        ex, ey = zip(*entries)
        ax_price.scatter(ex, ey, marker="^", color="green", s=100, zorder=5, label="Achat")
    if exits:
        sx, sy = zip(*exits)
        ax_price.scatter(sx, sy, marker="v", color="red", s=100, zorder=5, label="Vente")

    ax_price.set_title(
        f"{name}   —   RSI({rsi_period}) / EMA{short_period}(RSI) x EMA{long_period}(RSI)"
    )
    ax_price.legend(loc="upper left")
    ax_price.grid(alpha=0.3)

    ax_rsi.plot(rsi.index, rsi.values, color="#9467bd", linewidth=0.8, alpha=0.5, label=f"RSI({rsi_period})")
    ax_rsi.plot(ema_short.index, ema_short.values, color="#2ca02c", linewidth=1.2, label=f"EMA{short_period}(RSI) courte")
    ax_rsi.plot(ema_long.index, ema_long.values, color="#d62728", linewidth=1.2, label=f"EMA{long_period}(RSI) longue")
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel("RSI")
    ax_rsi.legend(loc="upper left", fontsize=8)
    ax_rsi.grid(alpha=0.3)

    fig.tight_layout()
    safe_name = "".join(c if c.isalnum() else "_" for c in name)
    path = charts_dir / f"{safe_name}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def run_grid_search(tickers: dict) -> pd.DataFrame:
    rsi_periods, ema_periods = list(RSI_PERIODS), list(EMA_PERIODS)
    ema_pairs = list(combinations(ema_periods, 2))  # (courte, longue), courte < longue
    n_combos = len(rsi_periods) * len(ema_pairs)
    print(
        f"{n_combos} combinaisons à tester "
        f"({len(rsi_periods)} périodes RSI x {len(ema_pairs)} paires d'EMA courte/longue)...\n"
    )

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
    for rsi_period, (short_period, long_period) in product(rsi_periods, ema_pairs):
        closed_sum = latent_sum = 0.0
        n_closed = n_wins = 0

        for name, closes in tickers.items():
            ema_short = ema_cache[(rsi_period, short_period)][name]
            ema_long = ema_cache[(rsi_period, long_period)][name]
            for t in find_trades(name, closes, ema_short, ema_long):
                if t["Statut"] == "Clôturé":
                    closed_sum += t["Rendement (%)"]
                    n_closed += 1
                    n_wins += t["Rendement (%)"] > 0
                else:
                    latent_sum += t["Rendement (%)"]

        results.append(
            {
                "Période RSI": rsi_period,
                "EMA courte": short_period,
                "EMA longue": long_period,
                "Nb trades clôturés": n_closed,
                "Somme rendements clôturés (%)": round(closed_sum, 2),
                "Somme rendements latents (%)": round(latent_sum, 2),
                "Total combiné (%)": round(closed_sum + latent_sum, 2),
                "Taux de réussite (%)": round(n_wins / n_closed * 100, 1) if n_closed else None,
            }
        )

    return pd.DataFrame(results).sort_values("Total combiné (%)", ascending=False).reset_index(drop=True)


def run_full_backtest(tickers: dict, rsi_period: int, short_period: int, long_period: int, charts_dir: Path):
    """Rejoue la meilleure combinaison en détail : trades, résumé, graphiques.

    Le total est une somme brute de tous les rendements individuels de
    trades, exactement comme dans le classement du grid search, pour
    que les deux nombres coïncident toujours sur une même combinaison.
    """
    all_trades = []
    summary_rows = []

    for name, closes in tickers.items():
        rsi = compute_rsi(closes, rsi_period)
        ema_short = compute_rsi_ema(rsi, short_period)
        ema_long = compute_rsi_ema(rsi, long_period)
        trades = find_trades(name, closes, ema_short, ema_long)
        all_trades.extend(trades)

        closed = [t for t in trades if t["Statut"] == "Clôturé"]
        open_trade = next((t for t in trades if t["Statut"] != "Clôturé"), None)

        closed_sum = sum(t["Rendement (%)"] for t in closed)
        wins = sum(1 for t in closed if t["Rendement (%)"] > 0)

        summary_rows.append(
            {
                "Valeur": name,
                "Nb trades clôturés": len(closed),
                "Somme rendements clôturés (%)": round(closed_sum, 2) if closed else None,
                "Taux de réussite (%)": round(wins / len(closed) * 100, 1) if closed else None,
                "Position ouverte": open_trade is not None,
                "Rendement latent position ouverte (%)": open_trade["Rendement (%)"] if open_trade else None,
            }
        )

        plot_ticker(name, closes, rsi, ema_short, ema_long, trades, charts_dir, rsi_period, short_period, long_period)

    summary = pd.DataFrame(summary_rows).set_index("Valeur").sort_values(
        "Somme rendements clôturés (%)", ascending=False
    )

    total_closed = summary["Somme rendements clôturés (%)"].sum(skipna=True)
    total_latent = summary["Rendement latent position ouverte (%)"].sum(skipna=True)
    total_row = pd.DataFrame(
        [
            {
                "Nb trades clôturés": int(summary["Nb trades clôturés"].sum()),
                "Somme rendements clôturés (%)": round(total_closed, 2),
                "Taux de réussite (%)": None,
                "Position ouverte": int(summary["Position ouverte"].sum()),
                "Rendement latent position ouverte (%)": round(total_latent, 2),
            }
        ],
        index=[f"TOTAL (somme des {len(summary)} valeurs)"],
    )
    summary = pd.concat([summary, total_row])

    trades_df = pd.DataFrame(all_trades)
    if not trades_df.empty:
        trades_df["Date entrée"] = pd.to_datetime(trades_df["Date entrée"]).dt.date
        trades_df["Date sortie"] = pd.to_datetime(trades_df["Date sortie"]).dt.date

    return summary, trades_df, total_closed, total_latent


def main() -> None:
    csv_path = latest_prices_csv()
    print(f"Données utilisées : {csv_path.name}")

    prices = pd.read_csv(csv_path, index_col=0, parse_dates=True, encoding="utf-8-sig")
    tickers = {name: prices[name].dropna() for name in prices.columns}
    print(f"{len(prices)} clôtures x {len(tickers)} valeurs")

    grid = run_grid_search(tickers)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_csv = OUTPUT_DIR / f"optimisation_macd_rsi_{timestamp}.csv"
    grid.to_csv(grid_csv, index=False, encoding="utf-8-sig")

    print("Top 15 des combinaisons (classées par rendement total combiné) :\n")
    print(grid.head(15).to_string(index=False))

    best = grid.iloc[0]
    rsi_period = int(best["Période RSI"])
    short_period = int(best["EMA courte"])
    long_period = int(best["EMA longue"])

    print(
        "\nMeilleure configuration trouvée :\n"
        f"- Période RSI      : {rsi_period}\n"
        f"- EMA courte(RSI)  : {short_period}\n"
        f"- EMA longue(RSI)  : {long_period}\n"
        f"- Trades clôturés  : {int(best['Nb trades clôturés'])} "
        f"(taux de réussite {best['Taux de réussite (%)']}%)\n"
        f"- Rendement total combiné (toutes valeurs, clôturé + latent) : "
        f"{best['Total combiné (%)']:+.2f} %"
    )
    print(f"\nClassement complet ({len(grid)} combinaisons) : {grid_csv.name}")
    print(
        "\n⚠️ Avertissement : cette combinaison est la meilleure SUR CE PASSÉ "
        "précis (mêmes 5 ans, mêmes valeurs qui ont servi à la choisir). Avec "
        f"{len(grid)} combinaisons testées, le risque de surapprentissage est "
        "réel — à valider sur une autre période ou un autre échantillon de "
        "valeurs avant d'y faire confiance en réel."
    )

    print("\nGénération du détail et des graphiques pour la meilleure configuration...")
    charts_dir = OUTPUT_DIR / f"graphiques_macd_rsi_meilleure_config_{timestamp}"
    charts_dir.mkdir(exist_ok=True)

    summary, trades_df, total_closed, total_latent = run_full_backtest(
        tickers, rsi_period, short_period, long_period, charts_dir
    )

    print("\nRésumé par valeur (meilleure configuration) :\n")
    print(summary)
    print(f"\nSomme des rendements des trades clôturés (toutes valeurs) : {total_closed:+.2f} %")
    print(f"Somme des rendements latents (positions encore ouvertes)  : {total_latent:+.2f} %")
    print(f"Total combiné (clôturés + latents)                        : {total_closed + total_latent:+.2f} %")

    trades_csv = OUTPUT_DIR / f"trades_macd_rsi_meilleure_config_{timestamp}.csv"
    trades_xlsx = OUTPUT_DIR / f"trades_macd_rsi_meilleure_config_{timestamp}.xlsx"
    trades_df.to_csv(trades_csv, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(trades_xlsx) as writer:
        summary.to_excel(writer, sheet_name="Résumé")
        trades_df.to_excel(writer, sheet_name="Trades", index=False)

    n_charts = len(list(charts_dir.glob("*.png")))
    print(f"\nDétail des trades : {trades_csv.name} / {trades_xlsx.name}")
    print(f"Graphiques ({n_charts} fichiers PNG, paramètres affichés en titre) : {charts_dir.name}/")


if __name__ == "__main__":
    main()
