"""
Download a station's full GHCN-Daily record in ONE request (no API token).

NOAA publishes every GHCN-Daily station as a single gzip CSV:
    https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/<ID>.csv.gz

Compared with the CDO web API (one request per year and per 1000 records,
10,000 requests/day, token required) this is what makes it practical to pull
records back to 1900: a century of data for a station is ~1 MB and one request.

Output matches the CSV the rest of the app expects:
    DATE, PRCP (mm), TMAX (deg C), TMIN (deg C)
"""
import io

import pandas as pd
import requests

BY_STATION_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/{sid}.csv.gz"

_COLS = ["ID", "DATE", "ELEMENT", "VALUE", "M_FLAG", "Q_FLAG", "S_FLAG", "OBS_TIME"]
_ELEMENTS = ("PRCP", "TMAX", "TMIN")


def bare_station_id(station_id: str) -> str:
    """'GHCND:USC00290600' -> 'USC00290600'."""
    return station_id.split(":")[-1].strip()


def download_ghcnd_station(station_id: str, start_date=None, end_date=None,
                           timeout: int = 120) -> pd.DataFrame:
    """Return DATE/PRCP/TMAX/TMIN for one station between two dates (inclusive).

    Values that failed NOAA's quality checks (non-blank quality flag) are dropped.
    Raises requests.HTTPError / requests.RequestException on network problems so
    the caller can fall back to another source.
    """
    sid = bare_station_id(station_id)
    r = requests.get(BY_STATION_URL.format(sid=sid), timeout=timeout)
    r.raise_for_status()

    raw = pd.read_csv(io.BytesIO(r.content), compression="gzip", header=None,
                      names=_COLS, usecols=["DATE", "ELEMENT", "VALUE", "Q_FLAG"],
                      dtype={"DATE": str, "ELEMENT": str, "Q_FLAG": str})

    raw = raw[raw["ELEMENT"].isin(_ELEMENTS) & raw["Q_FLAG"].isna()]
    if raw.empty:
        return pd.DataFrame(columns=["DATE", *_ELEMENTS])

    wide = (raw.pivot_table(index="DATE", columns="ELEMENT", values="VALUE",
                            aggfunc="first")
               .reset_index())
    wide.columns.name = None
    wide["DATE"] = pd.to_datetime(wide["DATE"], format="%Y%m%d")

    for col in _ELEMENTS:
        if col not in wide.columns:
            wide[col] = float("nan")
        wide[col] = wide[col] / 10.0        # tenths of mm / tenths of deg C -> mm / deg C

    if start_date is not None:
        wide = wide[wide["DATE"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        wide = wide[wide["DATE"] <= pd.to_datetime(end_date)]

    return wide[["DATE", *_ELEMENTS]].sort_values("DATE").reset_index(drop=True)
