import requests
import pandas as pd


def download_noaa_yearly(station_id: str, start_year: int, end_year: int,
                          token: str = "") -> pd.DataFrame:
    """
    Download NOAA GHCND daily data (PRCP, TMAX, TMIN) year by year.

    Parameters
    ----------
    station_id : str   e.g. "GHCND:USW00023050"
    start_year : int
    end_year   : int
    token      : str   NOAA CDO API token (https://www.ncdc.noaa.gov/cdo-web/token)
                       Required: no token is bundled with the code.
    """

    _token = token.strip() if token else ""
    if not _token:
        raise ValueError("A NOAA CDO API token is required "
                         "(https://www.ncdc.noaa.gov/cdo-web/token).")

    url     = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
    headers = {"token": _token}
    frames  = []

    for year in range(start_year, end_year + 1):

        params = {
            "datasetid": "GHCND",
            "stationid": station_id,
            "startdate": f"{year}-01-01",
            "enddate":   f"{year}-12-31",
            "limit":     1000,
            "units":     "metric",
        }
        offset = 1

        while True:
            params["offset"] = offset
            try:
                r    = requests.get(url, params=params, headers=headers, timeout=30)
                data = r.json()
            except Exception:
                break

            if "results" not in data:
                break

            frames.append(pd.DataFrame(data["results"]))

            if len(data["results"]) < 1000:
                break

            offset += 1000

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Pivot to wide format: one row per date, one column per datatype
    df = df.pivot_table(
        index="date", columns="datatype",
        values="value", aggfunc="first"
    ).reset_index()

    df.rename(columns={"date": "DATE"}, inplace=True)

    # Guarantee columns exist even if NOAA returned no data for that variable
    for col in ["PRCP", "TMAX", "TMIN"]:
        if col not in df.columns:
            df[col] = None

    return df[["DATE", "PRCP", "TMAX", "TMIN"]]
