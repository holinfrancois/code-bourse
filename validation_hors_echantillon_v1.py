"""Validation hors échantillon (walk-forward) de la stratégie RSI +
filtre de tendance + stop suiveur : on optimise les paramètres sur la
première partie de l'historique (période d'ENTRAÎNEMENT, 60% des
séances) puis on rejoue la SEULE combinaison retenue sur la partie
suivante (période de TEST, 40% restants), jamais vue par l'optimiseur.
C'est le vrai test du surapprentissage : si la performance s'effondre
en test, la combinaison ne captait pas un edge réel, juste une
particularité du passé sur lequel elle a été choisie.

Les indicateurs (RSI, EMA du RSI, EMA de tendance) sont calculés sur
tout l'historique disponible pour ne pas perdre les 20 à 80 premières
séances de la période de test à leur propre mise en route — seule la
fenêtre d'ENTRÉE/SORTIE des trades est restreinte à la période
considérée (entraînement ou test), donc aucune information de la
période de test ne fuite dans le choix des paramètres.

Stratégie testée (identique à optimisation_rsi_v11), long-only, une
position à la fois par valeur :
- Achat : le cours de clôture est AU-DESSUS de sa propre moyenne
  mobile exponentielle de tendance (filtre remplaçant le seuil de
  niveau RSI des versions précédentes) ET la moyenne mobile
  exponentielle COURTE du RSI passe au-dessus de la moyenne mobile
  exponentielle LONGUE du RSI.
- Vente : stop suiveur classique. On garde le plus haut cours de
  clôture DEPUIS L'ENTRÉE (sans limite de durée, monotone croissant —
  il ne redescend jamais) ; dès que la clôture retombe de plus de X %
  sous ce plus haut, on vend.

  Note : sur les versions précédentes, la stratégie restait déficiente
  dans les longues tendances baissières — le stop suiveur limite la
  casse une fois en position, mais rien n'empêchait d'entrer à
  contre-tendance en premier lieu. Le filtre de tendance sur le prix
  (plutôt qu'un niveau de RSI, qui ne dit rien de la tendance de fond)
  vise à n'acheter que lorsque le cours est structurellement orienté à
  la hausse.

Balaie :
- Période du RSI            : 6 à 16 par pas de 2
- EMA COURTE du RSI         : 2 à 8 par pas de 2
- EMA LONGUE du RSI         : 4 à 18 par pas de 2 (seules les paires où
  la courte est bien inférieure à la longue sont testées)
- EMA de tendance (sur le cours, pas le RSI) : 20 à 80 semaines par pas
  de 5 (plage relevée par rapport à la version précédente qui partait
  de 3 semaines : une EMA aussi courte colle de trop près au cours et
  ne filtre pas vraiment une tendance de fond — l'optimiseur la
  retenait alors qu'elle ne jouait pas son rôle de filtre)
- Perte du stop suiveur     : 6 % à 20 % par pas de 2
(16 224 combinaisons)

La sélection des paramètres, sur la période d'ENTRAÎNEMENT seulement,
se fait par RENDEMENT MÉDIAN PAR VALEUR (et non par la somme totale) :
un total brut peut être tiré vers le haut par 2-3 grosses valeurs
pendant que la majorité des autres perdent — la médiane reflète la
performance typique sur l'ensemble de l'échantillon, plus robuste.

La combinaison retenue est ensuite rejouée en détail SUR LES DEUX
PÉRIODES séparément (entraînement, puis test hors échantillon) :
trades, résumé avec ligne TOTAL, et un graphique par valeur pour
chacune. Le script termine par une comparaison directe rendement
médian / % de valeurs profitables entraînement vs test — l'écart entre
les deux est la vraie mesure du surapprentissage.

⚠️ Même hors échantillon, un split unique (une seule frontière
entraînement/test) reste un test partiel : une performance qui tient
sur ce découpage précis n'est pas une garantie absolue, mais un
effondrement net entre entraînement et test est déjà un signal fort
que la combinaison ne capte pas un edge réel.

Sorties, à côté du script :
- optimisation_stop_rsi_entrainement_<horodatage>.csv : classement
  complet des 16 224 combinaisons sur la période d'entraînement.
- trades_entrainement_<horodatage>.csv/.xlsx et
  graphiques_entrainement_<horodatage>/ : détail et graphiques sur la
  période d'entraînement.
- trades_test_horsech_<horodatage>.csv/.xlsx et
  graphiques_test_horsech_<horodatage>/ : détail et graphiques sur la
  période de test hors échantillon (jamais vue par l'optimiseur).
"""

from datetime import datetime
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RSI_PERIODS = range(6, 17, 2)
EMA_SHORT_PERIODS = range(2, 9, 2)
EMA_LONG_PERIODS = range(4, 19, 2)
TREND_EMA_PERIODS = range(20, 81, 5)
STOP_LOSS_PCTS = range(6, 21, 2)

TRAIN_FRACTION = 0.6  # 60% entraînement / 40% test hors échantillon

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


def compute_price_ema(closes: pd.Series, period: int) -> pd.Series:
    """Moyenne mobile exponentielle appliquée au cours (filtre de tendance)."""
    return closes.ewm(span=period, adjust=False).mean()


def find_trades(
    name: str,
    closes: pd.Series,
    ema_short: pd.Series,
    ema_long: pd.Series,
    trend_ema: pd.Series,
    stop_loss_pct: float,
) -> tuple:
    """Rejoue la stratégie et renvoie (détail de chaque trade, niveau du
    stop suiveur dans le temps — utile pour le tracer sur le graphique).

    Stop suiveur classique : pendant qu'on est en position, on garde le
    plus haut cours de clôture DEPUIS L'ENTRÉE (jamais borné, jamais
    décroissant) et on vend dès que la clôture repasse sous ce plus
    haut réduit de `stop_loss_pct` %.
    """
    in_position = False
    entry_date = entry_price = None
    trailing_high = None
    loss_frac = 1 - stop_loss_pct / 100
    trades = []
    stop_level_series = pd.Series(index=closes.index, dtype=float)

    for dt, price in closes.items():
        es = ema_short.get(dt)
        el = ema_long.get(dt)
        tr = trend_ema.get(dt)
        if pd.isna(es) or pd.isna(el) or pd.isna(tr):
            continue

        if not in_position:
            if price > tr and es > el:
                in_position = True
                entry_date, entry_price = dt, price
                trailing_high = price
        else:
            trailing_high = max(trailing_high, price)
            stop_level = trailing_high * loss_frac
            stop_level_series[dt] = stop_level
            if price < stop_level:
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

    return trades, stop_level_series


def simulate_fast(closes_arr, trend_arr, es_arr, el_arr, stop_loss_pct: float):
    """Même logique que find_trades, mais sur tableaux numpy et sans
    construire d'objets détaillés — utilisé pour le grid search, où l'on
    a besoin d'aller vite sur des dizaines de milliers de combinaisons.
    Renvoie (somme rendements clôturés, nb clôturés, nb gagnants, latent).
    """
    in_position = False
    entry_price = 0.0
    trailing_high = 0.0
    loss_frac = 1 - stop_loss_pct / 100
    closed_sum = 0.0
    n_closed = 0
    n_wins = 0
    latent = 0.0

    n = len(closes_arr)
    for i in range(n):
        tr = trend_arr[i]
        es = es_arr[i]
        el = el_arr[i]
        if tr != tr or es != es or el != el:  # plus rapide que pd.isna/np.isnan ici
            continue
        price = closes_arr[i]

        if not in_position:
            if price > tr and es > el:
                in_position = True
                entry_price = price
                trailing_high = price
        else:
            if price > trailing_high:
                trailing_high = price
            if price < trailing_high * loss_frac:
                ret = (price / entry_price - 1) * 100
                closed_sum += ret
                n_closed += 1
                if ret > 0:
                    n_wins += 1
                in_position = False

    if in_position:
        latent = (closes_arr[-1] / entry_price - 1) * 100

    return closed_sum, n_closed, n_wins, latent


def plot_ticker(
    name: str,
    closes: pd.Series,
    rsi: pd.Series,
    ema_short: pd.Series,
    ema_long: pd.Series,
    trend_ema: pd.Series,
    trades: list,
    stop_level_series: pd.Series,
    charts_dir: Path,
    rsi_period: int,
    short_period: int,
    long_period: int,
    trend_period: int,
    stop_loss_pct: float,
) -> Path:
    fig, (ax_price, ax_rsi) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_price.plot(closes.index, closes.values, color="#1f77b4", linewidth=1.2, label="Clôture")
    ax_price.plot(
        trend_ema.index, trend_ema.values,
        color="#ff7f0e", linewidth=1, label=f"EMA{trend_period} tendance",
    )
    if stop_level_series.notna().any():
        ax_price.plot(
            stop_level_series.index, stop_level_series.values,
            color="#d62728", linewidth=1, linestyle="--", label="Stop suiveur",
        )

    entries = [(t["Date entrée"], t["Prix entrée"]) for t in trades]
    exits = [(t["Date sortie"], t["Prix sortie"]) for t in trades if t["Date sortie"] is not None]

    if entries:
        ex, ey = zip(*entries)
        ax_price.scatter(ex, ey, marker="^", color="green", s=100, zorder=5, label="Achat")
    if exits:
        sx, sy = zip(*exits)
        ax_price.scatter(sx, sy, marker="v", color="red", s=100, zorder=5, label="Vente (stop)")

    ax_price.set_title(
        f"{name}   —   RSI({rsi_period}) / EMA{short_period}(RSI) x EMA{long_period}(RSI) "
        f"/ tendance EMA{trend_period}  /  stop suiveur {stop_loss_pct}% depuis l'entrée"
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


def run_grid_search(tickers: dict, split_idx: int) -> pd.DataFrame:
    """Grid search restreint à la période d'entraînement.

    `tickers` contient les cours COMPLETS (tout l'historique) : les
    indicateurs sont calculés dessus pour être correctement "chauffés",
    mais seules les `split_idx` premières séances de chaque tableau
    sont effectivement simulées, donc aucune séance de la période de
    test ne peut influencer le choix des paramètres.
    """
    rsi_periods = list(RSI_PERIODS)
    short_periods = list(EMA_SHORT_PERIODS)
    long_periods = list(EMA_LONG_PERIODS)
    trend_periods = list(TREND_EMA_PERIODS)
    stop_losses = list(STOP_LOSS_PCTS)
    ema_pairs = [(s, l) for s in short_periods for l in long_periods if s < l]
    n_combos = len(rsi_periods) * len(ema_pairs) * len(trend_periods) * len(stop_losses)
    print(
        f"{n_combos} combinaisons à tester "
        f"({len(rsi_periods)} périodes RSI x {len(ema_pairs)} paires d'EMA x "
        f"{len(trend_periods)} EMA de tendance x {len(stop_losses)} pertes stop) "
        "— environ 1 minute sur 30 valeurs...\n"
    )

    rsi_cache = {
        rsi_period: {name: compute_rsi(closes, rsi_period) for name, closes in tickers.items()}
        for rsi_period in rsi_periods
    }
    ema_periods = sorted(set(short_periods) | set(long_periods))
    ema_cache = {
        (rsi_period, ema_period): {
            name: compute_rsi_ema(rsi_cache[rsi_period][name], ema_period) for name in tickers
        }
        for rsi_period in rsi_periods
        for ema_period in ema_periods
    }
    # L'EMA de tendance porte sur le cours, pas sur le RSI : indépendante
    # de rsi_period, on ne la calcule qu'une fois par (valeur, période).
    trend_cache = {
        trend_period: {name: compute_price_ema(closes, trend_period) for name, closes in tickers.items()}
        for trend_period in trend_periods
    }

    # Tableaux numpy pour la simulation rapide (évite les lookups pandas
    # par date, coûteux répétés des dizaines de milliers de fois).
    # Tronqués à split_idx : les indicateurs sont "chauffés" sur tout
    # l'historique (calculés ci-dessus sur `tickers` complet), mais seule
    # la période d'entraînement est effectivement rejouée ici.
    closes_arrs = {name: closes.values[:split_idx] for name, closes in tickers.items()}
    ema_arrs = {
        key: {name: s.values[:split_idx] for name, s in d.items()} for key, d in ema_cache.items()
    }
    trend_arrs = {
        key: {name: s.values[:split_idx] for name, s in d.items()} for key, d in trend_cache.items()
    }

    results = []
    done = 0
    last_rsi_period = None
    for rsi_period, (short_period, long_period), trend_period in product(rsi_periods, ema_pairs, trend_periods):
        if rsi_period != last_rsi_period:
            if last_rsi_period is not None:
                print(f"  ... {done}/{n_combos} combinaisons testées")
            print(f"Période RSI {rsi_period} ({rsi_periods.index(rsi_period) + 1}/{len(rsi_periods)})...")
            last_rsi_period = rsi_period

        for stop_loss_pct in stop_losses:
            closed_sum = latent_sum = 0.0
            n_closed = n_wins = 0
            per_ticker_closed = {}

            for name in tickers:
                c_sum, n_c, n_w, lat = simulate_fast(
                    closes_arrs[name],
                    trend_arrs[trend_period][name],
                    ema_arrs[(rsi_period, short_period)][name],
                    ema_arrs[(rsi_period, long_period)][name],
                    stop_loss_pct,
                )
                closed_sum += c_sum
                latent_sum += lat
                n_closed += n_c
                n_wins += n_w
                per_ticker_closed[name] = c_sum

            per_ticker_series = pd.Series(per_ticker_closed)
            median_return = per_ticker_series.median()
            pct_profitable = (per_ticker_series > 0).mean() * 100

            results.append(
                {
                    "Période RSI": rsi_period,
                    "EMA courte": short_period,
                    "EMA longue": long_period,
                    "EMA tendance": trend_period,
                    "Stop perte (%)": stop_loss_pct,
                    "Nb trades clôturés": n_closed,
                    "Rendement médian par valeur (%)": round(median_return, 2),
                    "Valeurs profitables (%)": round(pct_profitable, 1),
                    "Somme rendements clôturés (%)": round(closed_sum, 2),
                    "Somme rendements latents (%)": round(latent_sum, 2),
                    "Total combiné (%)": round(closed_sum + latent_sum, 2),
                    "Taux de réussite (%)": round(n_wins / n_closed * 100, 1) if n_closed else None,
                }
            )
            done += 1

    print(f"  ... {done}/{n_combos} combinaisons testées\n")
    return (
        pd.DataFrame(results)
        .sort_values(
            ["Rendement médian par valeur (%)", "Valeurs profitables (%)"], ascending=[False, False]
        )
        .reset_index(drop=True)
    )


def run_full_backtest(
    tickers_full: dict,
    period_slice: slice,
    rsi_period: int,
    short_period: int,
    long_period: int,
    trend_period: int,
    stop_loss_pct: float,
    charts_dir: Path,
):
    """Rejoue la combinaison en détail sur une sous-période : trades,
    résumé, graphiques.

    `tickers_full` contient les cours COMPLETS ; les indicateurs sont
    calculés dessus (donc correctement "chauffés"), puis tout est
    tronqué à `period_slice` (ex. slice(None, split_idx) pour
    l'entraînement, slice(split_idx, None) pour le test) avant d'être
    rejoué et tracé — pour l'entraînement comme pour le test, aucune
    séance hors de la période considérée n'entre dans le résultat.

    Le total est une somme brute de tous les rendements individuels de
    trades, exactement comme dans le classement du grid search, pour
    que les deux nombres coïncident toujours sur une même combinaison.
    """
    all_trades = []
    summary_rows = []

    for name, full_closes in tickers_full.items():
        rsi = compute_rsi(full_closes, rsi_period)
        ema_short = compute_rsi_ema(rsi, short_period)
        ema_long = compute_rsi_ema(rsi, long_period)
        trend_ema = compute_price_ema(full_closes, trend_period)

        closes = full_closes.iloc[period_slice]
        rsi = rsi.iloc[period_slice]
        ema_short = ema_short.iloc[period_slice]
        ema_long = ema_long.iloc[period_slice]
        trend_ema = trend_ema.iloc[period_slice]

        trades, stop_level_series = find_trades(
            name, closes, ema_short, ema_long, trend_ema, stop_loss_pct
        )
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

        plot_ticker(
            name, closes, rsi, ema_short, ema_long, trend_ema, trades, stop_level_series, charts_dir,
            rsi_period, short_period, long_period, trend_period, stop_loss_pct,
        )

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


def read_prices_csv(csv_path: Path) -> pd.DataFrame:
    """Lit le CSV de cours en tolérant les ré-enregistrements depuis Excel.

    Excel sur un poste français enregistre souvent un CSV modifié à la
    main en ANSI/Windows-1252 (et parfois avec un séparateur point-
    virgule) plutôt qu'en UTF-8/virgule comme l'export d'origine. On
    essaie UTF-8 d'abord, puis on retombe sur cp1252, avec détection
    automatique du séparateur dans les deux cas.
    """
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return pd.read_csv(csv_path, index_col=0, parse_dates=True, encoding=encoding, sep=None, engine="python")
        except UnicodeDecodeError:
            continue
    raise RuntimeError(
        f"Impossible de lire {csv_path.name} : encodage non reconnu (ni UTF-8 ni Windows-1252)."
    )


def export_period(tickers_full, period_slice, params, charts_dir, label, file_tag, timestamp):
    rsi_period, short_period, long_period, trend_period, stop_loss_pct = params
    summary, trades_df, total_closed, total_latent = run_full_backtest(
        tickers_full, period_slice, rsi_period, short_period, long_period, trend_period, stop_loss_pct, charts_dir
    )
    print(f"\nRésumé par valeur ({label}) :\n")
    print(summary)
    print(f"\nSomme des rendements des trades clôturés ({label}) : {total_closed:+.2f} %")
    print(f"Somme des rendements latents ({label})                : {total_latent:+.2f} %")
    print(f"Total combiné ({label})                                : {total_closed + total_latent:+.2f} %")

    trades_csv = OUTPUT_DIR / f"trades_{file_tag}_{timestamp}.csv"
    trades_xlsx = OUTPUT_DIR / f"trades_{file_tag}_{timestamp}.xlsx"
    trades_df.to_csv(trades_csv, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(trades_xlsx) as writer:
        summary.to_excel(writer, sheet_name="Résumé")
        trades_df.to_excel(writer, sheet_name="Trades", index=False)
    print(f"Détail des trades ({label}) : {trades_csv.name} / {trades_xlsx.name}")

    return summary


def main() -> None:
    csv_path = latest_prices_csv()
    print(f"Données utilisées : {csv_path.name}")

    prices = read_prices_csv(csv_path)
    tickers_full = {name: prices[name].dropna() for name in prices.columns}
    n = len(prices)
    split_idx = int(n * TRAIN_FRACTION)
    train_start, train_end = prices.index[0], prices.index[split_idx - 1]
    test_start, test_end = prices.index[split_idx], prices.index[-1]
    print(f"{n} clôtures x {len(tickers_full)} valeurs")
    print(f"Entraînement : {train_start.date()} -> {train_end.date()} ({split_idx} séances)")
    print(f"Test hors échantillon : {test_start.date()} -> {test_end.date()} ({n - split_idx} séances)\n")

    grid = run_grid_search(tickers_full, split_idx)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_csv = OUTPUT_DIR / f"optimisation_stop_rsi_entrainement_{timestamp}.csv"
    grid.to_csv(grid_csv, index=False, encoding="utf-8-sig")

    print(
        "Top 15 des combinaisons SUR L'ENTRAÎNEMENT SEULEMENT "
        "(classées par rendement médian par valeur) :\n"
    )
    print(grid.head(15).to_string(index=False))

    best = grid.iloc[0]
    rsi_period = int(best["Période RSI"])
    short_period = int(best["EMA courte"])
    long_period = int(best["EMA longue"])
    trend_period = int(best["EMA tendance"])
    stop_loss_pct = float(best["Stop perte (%)"])
    params = (rsi_period, short_period, long_period, trend_period, stop_loss_pct)

    print(
        "\nMeilleure configuration trouvée SUR L'ENTRAÎNEMENT :\n"
        f"- Période RSI               : {rsi_period}\n"
        f"- EMA courte(RSI)           : {short_period}\n"
        f"- EMA longue(RSI)           : {long_period}\n"
        f"- EMA de tendance (prix)    : {trend_period} semaines\n"
        f"- Perte du stop suiveur     : {stop_loss_pct} % (depuis l'entrée, sans limite de durée)\n"
        f"- Rendement médian par valeur (entraînement) : {best['Rendement médian par valeur (%)']:+.2f} %\n"
        f"- Valeurs profitables (entraînement)         : {best['Valeurs profitables (%)']:.1f} %"
    )
    print(f"\nClassement complet (entraînement, {len(grid)} combinaisons) : {grid_csv.name}")

    print("\nGénération du détail et des graphiques (entraînement puis test hors échantillon)...")
    charts_train = OUTPUT_DIR / f"graphiques_entrainement_{timestamp}"
    charts_test = OUTPUT_DIR / f"graphiques_test_horsech_{timestamp}"
    charts_train.mkdir(exist_ok=True)
    charts_test.mkdir(exist_ok=True)

    summary_train = export_period(
        tickers_full, slice(None, split_idx), params, charts_train, "ENTRAÎNEMENT", "entrainement", timestamp
    )
    summary_test = export_period(
        tickers_full, slice(split_idx, None), params, charts_test, "TEST HORS ÉCHANTILLON", "test_horsech", timestamp
    )

    median_train = summary_train["Somme rendements clôturés (%)"].iloc[:-1].median()
    median_test = summary_test["Somme rendements clôturés (%)"].iloc[:-1].median()
    pct_pos_train = (summary_train["Somme rendements clôturés (%)"].iloc[:-1] > 0).mean() * 100
    pct_pos_test = (summary_test["Somme rendements clôturés (%)"].iloc[:-1] > 0).mean() * 100

    print(
        "\n" + "=" * 70
        + "\nCOMPARAISON ENTRAÎNEMENT vs TEST HORS ÉCHANTILLON (même combinaison)\n"
        + "=" * 70
        + f"\nRendement médian par valeur : entraînement {median_train:+.2f} %  |  test {median_test:+.2f} %"
        + f"\nValeurs profitables         : entraînement {pct_pos_train:.1f} %  |  test {pct_pos_test:.1f} %"
    )
    if median_test < median_train / 2 or (median_train > 0 and median_test < 0):
        print(
            "\n⚠️ La performance s'effondre nettement hors échantillon : signe fort de "
            "surapprentissage — cette combinaison ne capte probablement pas un edge "
            "réel, juste une particularité de la période d'entraînement."
        )
    else:
        print(
            "\nLa performance se maintient dans un ordre de grandeur comparable hors "
            "échantillon — c'est plus rassurant, mais un seul split ne suffit pas à "
            "conclure définitivement (à confirmer sur d'autres découpages)."
        )

    for charts_dir, label in ((charts_train, "entraînement"), (charts_test, "test hors échantillon")):
        n_charts = len(list(charts_dir.glob("*.png")))
        print(f"Graphiques {label} ({n_charts} fichiers PNG) : {charts_dir.name}/")


if __name__ == "__main__":
    main()
