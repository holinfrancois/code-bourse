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


def fetch_closing_prices(tickers: dict, period: str = PERIOD) -> pd.DataFrame:
    """Télécharge les cours de clôture quotidiens pour les tickers donnés."""
    raw = yf.download(
        list(tickers.values()),
        period=period,
        interval=INTERVAL,
        auto_adjust=False,
        progress=False,
    )
    closes = raw["Close"]
    ticker_to_name = {ticker: name for name, ticker in tickers.items()}
    return closes.rename(columns=ticker_to_name).sort_index()


def main() -> None:
    prices = fetch_closing_prices(SRD_SAMPLE)

    print(f"Période demandée : {PERIOD} | Valeurs testées : {len(SRD_SAMPLE)}")
    print(f"Séances récupérées : {len(prices)}")
    if not prices.empty:
        print(f"Période couverte : {prices.index.min().date()} -> {prices.index.max().date()}")

    print("\nAperçu des 5 dernières séances (cours de clôture) :")
    print(prices.tail(5).round(2))

    print("\nValeurs manquantes par titre :")
    print(prices.isna().sum())


if __name__ == "__main__":
    main()
