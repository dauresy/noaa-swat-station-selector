import geopandas as gpd
import pandas as pd

def analyze_subbasins(stations, basin):

    st_in_sub = gpd.sjoin(stations, basin, predicate="within", how="left")
    sub_counts = st_in_sub.groupby("Subbasin").size().rename("n_stations").reset_index()

    utm_crs = basin.estimate_utm_crs()
    sub_utm = basin.to_crs(utm_crs).copy()
    st_utm = stations.to_crs(utm_crs).copy()

    sub_utm["centroid"] = sub_utm.geometry.centroid
    sub_utm["cx"] = sub_utm["centroid"].x
    sub_utm["cy"] = sub_utm["centroid"].y

    st_utm["sx"] = st_utm.geometry.x
    st_utm["sy"] = st_utm.geometry.y

    nearest_list = []

    for idx, row in sub_utm.iterrows():
        cx, cy = row["cx"], row["cy"]
        dists = ((st_utm["sx"] - cx)**2 + (st_utm["sy"] - cy)**2)**0.5
        min_idx = dists.idxmin()
        nearest_station = st_utm.loc[min_idx]

        nearest_list.append({
            "Subbasin": row["Subbasin"],
            "nearest_station_id": nearest_station["id"],
            "nearest_station_name": nearest_station["name"],
            "dist_km": dists[min_idx] / 1000,
            "sub_elev": row["Elev"],
            "station_elev": nearest_station["elevation"],
            "elev_diff": row["Elev"] - nearest_station["elevation"]
        })

    nearest_df = pd.DataFrame(nearest_list)

    return nearest_df, sub_counts
