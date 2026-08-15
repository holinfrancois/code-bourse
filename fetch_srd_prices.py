"""Récupère les cours de clôture quotidiens des valeurs éligibles au SRD.

Le SRD (Service de Règlement Différé) est un mécanisme d'Euronext Paris
permettant de différer le règlement d'un ordre en fin de mois. Il concerne
une liste de valeurs publiée par Euronext (essentiellement les plus grosses
capitalisations et les plus liquides de la cote parisienne).

Ce script est une première étape de vérification : il ne couvre pour
l'instant qu'un échantillon d'une dizaine de valeurs SRD parmi les plus
connues (à remplacer par la liste officielle complète d'Euronext une fois
le principe validé) et affiche un aperçu des données pour confirmer que la
récupération via Yahoo Finance (yfinance) fonctionne correctement.
"""

import time
import traceback
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

# Échantillon de valeurs éligibles au SRD (Euronext Paris), avec leur
# ticker Yahoo Finance (suffixe .PA). À étendre avec la liste officielle
# complète du SRD publiée par Euronext.
SRD_SAMPLE = {
    "TotalEnergies": "TTE.PA",
    "LVMH": "MC.PA",
    "Sanofi": "SAN.PA",
    "L'Oréal": "OR.PA",
    "BNP Paribas": "BNP.PA",
    "Air Liquide": "AI.PA",
    "Schneider Electric": "SU.PA",
    "AXA": "CS.PA",
    "Airbus": "AIR.PA",
    "Danone": "BN.PA",
}

PERIOD = "3mo"
INTERVAL = "1d"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2
OUTPUT_DIR = Path(__file__).resolve().parent


def _download_closes(tickers: list, period: str) -> pd.DataFrame:
    raw = yf.download(
        tickers,
        period=period,
        interval=INTERVAL,
        auto_adjust=False,
        progress=False,
    )
    if raw.empty:
        return pd.DataFrame(columns=tickers)
    return raw["Close"]


def fetch_closing_prices(tickers: dict, period: str = PERIOD) -> pd.DataFrame:
    """Télécharge les cours de clôture quotidiens pour les tickers donnés.

    Certaines valeurs peuvent échouer ponctuellement (ex. cache SQLite de
    yfinance verrouillé) sans que les autres soient affectées : on retente
    ces valeurs isolément avant d'abandonner.
    """
    closes = _download_closes(list(tickers.values()), period)

    for attempt in range(1, MAX_RETRIES + 1):
        missing = [t for t in tickers.values() if t not in closes.columns or closes[t].isna().all()]
        if not missing:
            break
        print(f"Nouvelle tentative ({attempt}/{MAX_RETRIES}) pour : {', '.join(missing)}")
        time.sleep(RETRY_DELAY_SECONDS)
        retry_closes = _download_closes(missing, period)
        for ticker in missing:
            if ticker in retry_closes.columns:
                closes[ticker] = retry_closes[ticker]

    ticker_to_name = {ticker: name for name, ticker in tickers.items()}
    return closes.rename(columns=ticker_to_name).sort_index()


def save_outputs(prices: pd.DataFrame) -> None:
    """Enregistre les cours récupérés en CSV et Excel, à côté du script.

    Les deux formats sont sauvegardés indépendamment : un échec sur l'un
    (ex. openpyxl absent pour l'Excel) n'empêche pas l'autre d'être écrit,
    et l'erreur complète est affichée au lieu d'être avalée en silence.
    """
    stem = f"srd_cours_cloture_{date.today():%Y%m%d}"
    print(f"\nDossier de sortie : {OUTPUT_DIR}")

    csv_path = OUTPUT_DIR / f"{stem}.csv"
    try:
        prices.to_csv(csv_path, encoding="utf-8-sig")
        print(f"CSV écrit : {csv_path} (existe : {csv_path.exists()})")
    except Exception:
        print(f"ÉCHEC écriture CSV ({csv_path}) :")
        traceback.print_exc()

    xlsx_path = OUTPUT_DIR / f"{stem}.xlsx"
    try:
        prices.to_excel(xlsx_path, sheet_name="Cours clôture")
        print(f"Excel écrit : {xlsx_path} (existe : {xlsx_path.exists()})")
    except Exception:
        print(f"ÉCHEC écriture Excel ({xlsx_path}) :")
        traceback.print_exc()


def main() -> None:
    prices = fetch_closing_prices(SRD_SAMPLE)

    print(f"\nPériode demandée : {PERIOD} | Valeurs testées : {len(SRD_SAMPLE)}")
    print(f"Séances récupérées : {len(prices)}")
    if not prices.empty:
        print(f"Période couverte : {prices.index.min().date()} -> {prices.index.max().date()}")

    print("\nAperçu des 5 dernières séances (cours de clôture) :")
    print(prices.tail(5).round(2))

    print("\nValeurs manquantes par titre :")
    print(prices.isna().sum())

    save_outputs(prices)


if __name__ == "__main__":
    main()
