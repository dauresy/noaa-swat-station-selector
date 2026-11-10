import pandas as pd
import requests

from .download_ghcnd_station import download_ghcnd_station
from .download_noaa_yearly import download_noaa_yearly


def download_noaa_daily(station_id: str, start_date, end_date,
                         token: str = "") -> pd.DataFrame:
    """
    Download daily NOAA GHCND data (PRCP, TMAX, TMIN) for one station.

    The full record is first requested from NOAA's GHCN-Daily file server in a
    single request (no token, any start date back to the station's first
    observation, 1900 and earlier included). Only if that fails and a CDO token
    is available does it fall back to the CDO API, which needs one request per
    year and is limited to 10,000 requests a day.

    Parameters
    ----------
    station_id : str   e.g. "GHCND:USW00023050"
    start_date : str | date | datetime
    end_date   : str | date | datetime
    token      : str   optional NOAA CDO API token, used only for the fallback
    """
    try:
        return download_ghcnd_station(station_id, start_date, end_date)
    except requests.RequestException:
        if not (token and token.strip()):
            raise

    start_year = pd.to_datetime(start_date).year
    end_year   = pd.to_datetime(end_date).year

    df = download_noaa_yearly(station_id, start_year, end_year, token=token)

    if df.empty:
        return df

    df["DATE"] = pd.to_datetime(df["DATE"])
    mask = (
        (df["DATE"] >= pd.to_datetime(start_date)) &
        (df["DATE"] <= pd.to_datetime(end_date))
    )
    return df.loc[mask].reset_index(drop=True)
