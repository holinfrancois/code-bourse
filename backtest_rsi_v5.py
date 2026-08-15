"""Backtest d'une stratégie RSI + moyenne mobile du RSI sur les cours SRD
déjà récupérés, avec détail de chaque trade et graphique par valeur pour
vérification visuelle.

Stratégie long-only, une position à la fois par valeur :
- Entrée : le RSI hebdomadaire est sous 50 ET repasse au-dessus de sa
  propre moyenne mobile exponentielle 20 périodes.
- Sortie : le RSI hebdomadaire repasse sous cette même moyenne mobile.

Ce script réutilise le dernier export CSV produit par le script de
récupération des cours (fichier srd_cours_cloture_*.csv le plus récent
du dossier) plutôt que de retélécharger les données.

Sorties produites, à côté du script :
- trades_rsi_<horodatage>.csv/.xlsx : le détail de chaque trade (date et
  prix d'entrée/sortie, rendement, statut clôturé/en cours) et un
  résumé par valeur, qui distingue le rendement des trades clôturés du
  rendement latent d'une éventuelle position encore ouverte (sinon une
  perte latente importante sur une position ouverte passerait inaperçue).
- graphiques_rsi_<horodatage>/ : un PNG par valeur, cours + RSI, avec
  triangle vert à chaque achat et triangle rouge à chaque vente.

Le résumé se termine par une ligne TOTAL : somme simple (non pondérée)
des rendements de toutes les valeurs, séparément pour les trades
clôturés et pour les positions encore ouvertes.
"""

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RSI_PERIOD = 14
RSI_EMA_PERIOD = 20
ENTRY_RSI_MAX = 50

OUTPUT_DIR = Path(__file__).resolve().parent


def latest_prices_csv() -> Path:
    candidates = sorted(OUTPUT_DIR.glob("srd_cours_cloture_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            "Aucun fichier srd_cours_cloture_*.csv trouvé dans "
            f"{OUTPUT_DIR}. Lance d'abord le script de récupération des cours."
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


def compute_rsi_ema(rsi: pd.Series, period: int = RSI_EMA_PERIOD) -> pd.Series:
    """Moyenne mobile exponentielle appliquée au RSI lui-même."""
    return rsi.ewm(span=period, adjust=False).mean()


def find_trades(name: str, closes: pd.Series, rsi: pd.Series, rsi_ema: pd.Series) -> list:
    """Rejoue la stratégie et renvoie le détail de chaque trade.

    Modèle à une position à la fois : dès qu'on sort d'un trade, le RSI
    est par construction repassé sous sa moyenne mobile, donc la
    prochaine entrée exige mécaniquement qu'il repasse au-dessus — pas
    besoin de comparer explicitement au point précédent pour détecter
    le croisement.
    """
    in_position = False
    entry_date = entry_price = None
    trades = []

    for dt, price in closes.items():
        r = rsi.get(dt)
        e = rsi_ema.get(dt)
        if pd.isna(r) or pd.isna(e):
            continue
        if not in_position and r < ENTRY_RSI_MAX and r > e:
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
    name: str, closes: pd.Series, rsi: pd.Series, rsi_ema: pd.Series, trades: list, charts_dir: Path
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

    ax_price.set_title(name)
    ax_price.legend(loc="upper left")
    ax_price.grid(alpha=0.3)

    ax_rsi.plot(rsi.index, rsi.values, color="#9467bd", linewidth=1, label="RSI")
    ax_rsi.plot(
        rsi_ema.index, rsi_ema.values, color="#ff7f0e", linewidth=1, label=f"EMA{RSI_EMA_PERIOD}(RSI)"
    )
    ax_rsi.axhline(ENTRY_RSI_MAX, color="grey", linestyle="--", linewidth=0.8)
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


def main() -> None:
    csv_path = latest_prices_csv()
    print(f"Données utilisées : {csv_path.name}")

    prices = pd.read_csv(csv_path, index_col=0, parse_dates=True, encoding="utf-8-sig")
    print(f"{len(prices)} clôtures x {len(prices.columns)} valeurs\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    charts_dir = OUTPUT_DIR / f"graphiques_rsi_{timestamp}"
    charts_dir.mkdir(exist_ok=True)

    all_trades = []
    summary_rows = []

    for name in prices.columns:
        closes = prices[name].dropna()
        if closes.empty:
            continue
        rsi = compute_rsi(closes)
        rsi_ema = compute_rsi_ema(rsi)
        trades = find_trades(name, closes, rsi, rsi_ema)
        all_trades.extend(trades)

        closed = [t for t in trades if t["Statut"] == "Clôturé"]
        open_trade = next((t for t in trades if t["Statut"] != "Clôturé"), None)

        total_return = 1.0
        for t in closed:
            total_return *= 1 + t["Rendement (%)"] / 100
        total_return -= 1
        wins = sum(1 for t in closed if t["Rendement (%)"] > 0)

        summary_rows.append(
            {
                "Valeur": name,
                "Nb trades clôturés": len(closed),
                "Rendement trades clôturés (%)": round(total_return * 100, 2) if closed else None,
                "Taux de réussite (%)": round(wins / len(closed) * 100, 1) if closed else None,
                "Position ouverte": open_trade is not None,
                "Rendement latent position ouverte (%)": open_trade["Rendement (%)"] if open_trade else None,
            }
        )

        plot_ticker(name, closes, rsi, rsi_ema, trades, charts_dir)

    summary = pd.DataFrame(summary_rows).set_index("Valeur").sort_values(
        "Rendement trades clôturés (%)", ascending=False
    )

    total_closed = summary["Rendement trades clôturés (%)"].sum(skipna=True)
    total_latent = summary["Rendement latent position ouverte (%)"].sum(skipna=True)
    total_row = pd.DataFrame(
        [
            {
                "Nb trades clôturés": int(summary["Nb trades clôturés"].sum()),
                "Rendement trades clôturés (%)": round(total_closed, 2),
                "Taux de réussite (%)": None,
                "Position ouverte": int(summary["Position ouverte"].sum()),
                "Rendement latent position ouverte (%)": round(total_latent, 2),
            }
        ],
        index=[f"TOTAL (somme des {len(summary)} valeurs)"],
    )
    summary = pd.concat([summary, total_row])

    print("Résumé par valeur :\n")
    print(summary)
    print(f"\nSomme des rendements des trades clôturés (toutes valeurs) : {total_closed:+.2f} %")
    print(f"Somme des rendements latents (positions encore ouvertes)  : {total_latent:+.2f} %")
    print(f"Total combiné (clôturés + latents)                        : {total_closed + total_latent:+.2f} %")

    trades_df = pd.DataFrame(all_trades)
    if not trades_df.empty:
        trades_df["Date entrée"] = pd.to_datetime(trades_df["Date entrée"]).dt.date
        trades_df["Date sortie"] = pd.to_datetime(trades_df["Date sortie"]).dt.date

    n_closed = int((trades_df["Statut"] == "Clôturé").sum()) if not trades_df.empty else 0
    print(f"\n{len(trades_df)} trade(s) au total, dont {n_closed} clôturé(s) :\n")
    print(trades_df.to_string(index=False))

    trades_csv = OUTPUT_DIR / f"trades_rsi_{timestamp}.csv"
    trades_xlsx = OUTPUT_DIR / f"trades_rsi_{timestamp}.xlsx"
    trades_df.to_csv(trades_csv, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(trades_xlsx) as writer:
        summary.to_excel(writer, sheet_name="Résumé")
        trades_df.to_excel(writer, sheet_name="Trades", index=False)

    n_charts = len(list(charts_dir.glob("*.png")))
    print(f"\nDétail des trades : {trades_csv.name} / {trades_xlsx.name}")
    print(f"Graphiques ({n_charts} fichiers PNG) : {charts_dir.name}/")


if __name__ == "__main__":
    main()
