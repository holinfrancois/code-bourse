"""Optimisation des paramètres de la stratégie RSI / EMA(RSI) sur les
cours SRD déjà récupérés (recherche exhaustive / grid search), puis
génération automatique du détail des trades et des graphiques pour la
meilleure combinaison trouvée.

Balaie :
- Période du RSI        : 6 à 16 par pas de 2
- Période de l'EMA(RSI)  : 4 à 18 par pas de 2
- Niveau d'entrée du RSI : 30 à 60 par pas de 5
(336 combinaisons)

Stratégie testée pour chaque combinaison :
- Achat : le RSI est sous le niveau d'entrée ET repasse au-dessus de sa
  propre EMA.
- Vente : le RSI repasse sous cette EMA.

Pour chaque combinaison, on calcule la somme des rendements (trades
clôturés + rendement latent des positions encore ouvertes) sur
l'ensemble des valeurs du dernier fichier de cours récupéré, puis on
classe les combinaisons par performance.

La meilleure combinaison est ensuite rejouée en détail (comme
backtest_rsi) : trades, résumé avec ligne TOTAL, et un graphique par
valeur — les graphiques affichent en titre les paramètres retenus
(période RSI, période EMA, seuil d'entrée) pour qu'on sache toujours à
quelle configuration ils correspondent.

⚠️ Avertissement surapprentissage (overfitting) : 336 combinaisons
testées sur seulement ~5 ans d'historique et une trentaine de valeurs,
c'est un terrain propice au surapprentissage — la meilleure combinaison
sur ce passé précis n'a aucune garantie de bien se comporter à l'avenir.
À confirmer sur une autre période ou un autre échantillon de valeurs
avant d'y faire confiance en réel.

Sorties, à côté du script :
- optimisation_rsi_<horodatage>.csv : le classement complet des 336
  combinaisons.
- trades_rsi_meilleure_config_<horodatage>.csv/.xlsx : détail des
  trades et résumé (avec TOTAL) pour la meilleure combinaison.
- graphiques_rsi_meilleure_config_<horodatage>/ : un PNG par valeur
  pour la meilleure combinaison, paramètres affichés en titre.
"""

from datetime import datetime
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def find_trades(name: str, closes: pd.Series, rsi: pd.Series, rsi_ema: pd.Series, entry_max: int) -> list:
    """Rejoue la stratégie et renvoie le détail de chaque trade."""
    in_position = False
    entry_date = entry_price = None
    trades = []

    for dt, price in closes.items():
        r = rsi.get(dt)
        e = rsi_ema.get(dt)
        if pd.isna(r) or pd.isna(e):
            continue
        if not in_position and r < entry_max and r > e:
            in_position = True
            entry_date, entry_price = dt, price
        elif in_position and r < e:
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
    rsi_ema: pd.Series,
    trades: list,
    charts_dir: Path,
    rsi_period: int,
    ema_period: int,
    entry_max: int,
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
        f"{name}   —   RSI({rsi_period}) / EMA{ema_period}(RSI) / seuil d'entrée {entry_max}"
    )
    ax_price.legend(loc="upper left")
    ax_price.grid(alpha=0.3)

    ax_rsi.plot(rsi.index, rsi.values, color="#9467bd", linewidth=1, label=f"RSI({rsi_period})")
    ax_rsi.plot(
        rsi_ema.index, rsi_ema.values, color="#ff7f0e", linewidth=1, label=f"EMA{ema_period}(RSI)"
    )
    ax_rsi.axhline(entry_max, color="grey", linestyle="--", linewidth=0.8, label=f"Seuil {entry_max}")
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
    rsi_periods, ema_periods, entry_levels = list(RSI_PERIODS), list(EMA_PERIODS), list(ENTRY_LEVELS)
    n_combos = len(rsi_periods) * len(ema_periods) * len(entry_levels)
    print(
        f"{n_combos} combinaisons à tester "
        f"({len(rsi_periods)} périodes RSI x {len(ema_periods)} périodes EMA "
        f"x {len(entry_levels)} niveaux d'entrée)...\n"
    )

    # RSI et EMA(RSI) mis en cache par (valeur, période) pour ne pas les
    # recalculer à chaque niveau d'entrée testé.
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
            for t in find_trades(name, closes, rsi, rsi_ema, entry_level):
                if t["Statut"] == "Clôturé":
                    closed_sum += t["Rendement (%)"]
                    n_closed += 1
                    n_wins += t["Rendement (%)"] > 0
                else:
                    latent_sum += t["Rendement (%)"]

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

    return pd.DataFrame(results).sort_values("Total combiné (%)", ascending=False).reset_index(drop=True)


def run_full_backtest(tickers: dict, rsi_period: int, ema_period: int, entry_max: int, charts_dir: Path):
    """Rejoue la meilleure combinaison en détail : trades, résumé, graphiques.

    Le total (ligne TOTAL et sommes affichées) est une somme brute de
    tous les rendements individuels de trades, exactement comme dans le
    classement du grid search — pas un rendement composé par valeur —
    pour que les deux nombres coïncident toujours sur une même
    combinaison.
    """
    all_trades = []
    summary_rows = []

    for name, closes in tickers.items():
        rsi = compute_rsi(closes, rsi_period)
        rsi_ema = compute_rsi_ema(rsi, ema_period)
        trades = find_trades(name, closes, rsi, rsi_ema, entry_max)
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

        plot_ticker(name, closes, rsi, rsi_ema, trades, charts_dir, rsi_period, ema_period, entry_max)

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
    grid_csv = OUTPUT_DIR / f"optimisation_rsi_{timestamp}.csv"
    grid.to_csv(grid_csv, index=False, encoding="utf-8-sig")

    print("Top 15 des combinaisons (classées par rendement total combiné) :\n")
    print(grid.head(15).to_string(index=False))

    best = grid.iloc[0]
    rsi_period = int(best["Période RSI"])
    ema_period = int(best["Période EMA(RSI)"])
    entry_max = int(best["Niveau entrée RSI"])

    print(
        "\nMeilleure configuration trouvée :\n"
        f"- Période RSI              : {rsi_period}\n"
        f"- Période EMA(RSI)          : {ema_period}\n"
        f"- Niveau d'entrée RSI       : {entry_max}\n"
        f"- Trades clôturés           : {int(best['Nb trades clôturés'])} "
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

    print(f"\nGénération du détail et des graphiques pour la meilleure configuration...")
    charts_dir = OUTPUT_DIR / f"graphiques_rsi_meilleure_config_{timestamp}"
    charts_dir.mkdir(exist_ok=True)

    summary, trades_df, total_closed, total_latent = run_full_backtest(
        tickers, rsi_period, ema_period, entry_max, charts_dir
    )

    print("\nRésumé par valeur (meilleure configuration) :\n")
    print(summary)
    print(f"\nSomme des rendements des trades clôturés (toutes valeurs) : {total_closed:+.2f} %")
    print(f"Somme des rendements latents (positions encore ouvertes)  : {total_latent:+.2f} %")
    print(f"Total combiné (clôturés + latents)                        : {total_closed + total_latent:+.2f} %")

    trades_csv = OUTPUT_DIR / f"trades_rsi_meilleure_config_{timestamp}.csv"
    trades_xlsx = OUTPUT_DIR / f"trades_rsi_meilleure_config_{timestamp}.xlsx"
    trades_df.to_csv(trades_csv, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(trades_xlsx) as writer:
        summary.to_excel(writer, sheet_name="Résumé")
        trades_df.to_excel(writer, sheet_name="Trades", index=False)

    n_charts = len(list(charts_dir.glob("*.png")))
    print(f"\nDétail des trades : {trades_csv.name} / {trades_xlsx.name}")
    print(f"Graphiques ({n_charts} fichiers PNG, paramètres affichés en titre) : {charts_dir.name}/")


if __name__ == "__main__":
    main()
