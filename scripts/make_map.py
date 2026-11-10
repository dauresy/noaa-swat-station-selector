import geopandas as gpd
import folium
from folium.plugins import MarkerCluster

def make_map(stations, basin, buffer_km):

    basin_union = basin.union_all()
    basin_single = gpd.GeoDataFrame(geometry=[basin_union], crs=4326)

    centroid = basin_single.geometry.union_all().centroid

    m = folium.Map(
        location=[centroid.y, centroid.x],
        zoom_start=7,
        tiles="OpenStreetMap"
    )

    folium.GeoJson(
        basin[[basin.geometry.name]].to_json(),
        name="Subbasins",
        style_function=lambda x: {"color": "#555", "weight": 1, "fillOpacity": 0.1}
    ).add_to(m)

    if buffer_km > 0:
        utm_crs = basin_single.estimate_utm_crs()
        basin_utm = basin_single.to_crs(utm_crs)
        buffer = basin_utm.buffer(buffer_km * 1000)
        buffer = gpd.GeoDataFrame(geometry=buffer, crs=utm_crs).to_crs(4326)

        folium.GeoJson(
            buffer.to_json(),
            name="Buffer",
            style_function=lambda x: {"color": "red", "weight": 2, "dashArray": "5,5"}
        ).add_to(m)

    cluster = MarkerCluster(name="Stations").add_to(m)

    for _, row in stations.iterrows():
        popup_html = (
            f"<b>{row['name']}</b><br>"
            f"{row['id']}<br>"
            f"Elevation: {row['elevation']} m<br>"
            f"Score: {row['score']:.2f}<br>"
        )

        folium.Marker(
            location=[row.latitude, row.longitude],
            popup=popup_html,
            icon=folium.Icon(color="blue")
        ).add_to(cluster)

    folium.LayerControl().add_to(m)

    return m
