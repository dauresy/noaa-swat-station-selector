import geopandas as gpd
import pandas as pd

from .select_stations import _fetch_all_stations


def download_noaa_stations(basin, buffer_km, token):
    """All GHCND stations in the basin's buffered bounding box (needs a CDO token)."""
    basin_union = basin.union_all()
    basin_single = gpd.GeoDataFrame(geometry=[basin_union], crs=4326)

    if buffer_km > 0:
        utm_crs = basin_single.estimate_utm_crs()
        basin_utm = basin_single.to_crs(utm_crs)
        basin_buffer = basin_utm.buffer(buffer_km * 1000)
        basin_buffer = gpd.GeoDataFrame(geometry=basin_buffer, crs=utm_crs).to_crs(4326)
    else:
        basin_buffer = basin_single

    west, south, east, north = basin_buffer.total_bounds
    return pd.DataFrame(_fetch_all_stations(f"{south},{west},{north},{east}", token))
