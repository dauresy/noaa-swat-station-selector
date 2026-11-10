import geopandas as gpd
import pandas as pd

def filter_and_score(stations, min_year, cal_start, cal_end, min_years, min_cover, min_dist_km):

    # ⭐ NORMALIZAR NOMBRES DE COLUMNAS DE FECHAS
    # mindate
    if "mindate" in stations.columns:
        stations["mindate"] = pd.to_datetime(stations["mindate"], errors="coerce")
    elif "minDate" in stations.columns:
        stations["mindate"] = pd.to_datetime(stations["minDate"], errors="coerce")
    elif "min_date" in stations.columns:
        stations["mindate"] = pd.to_datetime(stations["min_date"], errors="coerce")
    else:
        stations["mindate"] = pd.NaT

    # maxdate
    if "maxdate" in stations.columns:
        stations["maxdate"] = pd.to_datetime(stations["maxdate"], errors="coerce")
    elif "maxDate" in stations.columns:
        stations["maxdate"] = pd.to_datetime(stations["maxDate"], errors="coerce")
    elif "max_date" in stations.columns:
        stations["maxdate"] = pd.to_datetime(stations["max_date"], errors="coerce")
    else:
        stations["maxdate"] = pd.NaT

    # ⭐ CALCULAR SPAN DE AÑOS
    stations["span_years"] = (stations["maxdate"] - stations["mindate"]).dt.days / 365.25

    # ⭐ CREAR GEODATAFRAME
    g_st = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations.longitude, stations.latitude),
        crs=4326
    )

    # ⭐ FILTRO POR AÑO MÍNIMO
    g_st = g_st[g_st["maxdate"].dt.year >= min_year]

    # ⭐ COBERTURA EN PERÍODO DE CALIBRACIÓN
    cal_start = pd.to_datetime(cal_start)
    cal_end = pd.to_datetime(cal_end)
    cal_years = cal_end.year - cal_start.year + 1

    g_st["years_in_cal"] = g_st.apply(
        lambda r: max(0, min(r["maxdate"].year, cal_end.year) - max(r["mindate"].year, cal_start.year) + 1)
        if pd.notnull(r["mindate"]) and pd.notnull(r["maxdate"]) else 0,
        axis=1
    )

    g_st["cover_frac"] = g_st["years_in_cal"] / cal_years

    # ⭐ FILTROS
    g_st = g_st[g_st["cover_frac"] >= min_cover]
    g_st = g_st[g_st["span_years"] >= min_years]

    # ⭐ NORMALIZACIÓN DE RECENCIA
    if len(g_st) > 0:
        max_span = g_st["span_years"].max()
        min_year_rec = g_st["maxdate"].dt.year.min()
        max_year_rec = g_st["maxdate"].dt.year.max()

        g_st["recency_norm"] = (g_st["maxdate"].dt.year - min_year_rec) / (max_year_rec - min_year_rec)
    else:
        g_st["recency_norm"] = 0

    # ⭐ SCORE FINAL
    g_st["score"] = (
        0.4 * (g_st["span_years"] / g_st["span_years"].max()) +
        0.4 * g_st["cover_frac"] +
        0.2 * g_st["recency_norm"]
    )

    g_st = g_st.sort_values("score", ascending=False)

    # ⭐ THINNING POR DISTANCIA
    st_utm = g_st.to_crs(g_st.estimate_utm_crs())
    selected_idx = []
    used = set()
    min_dist_m = min_dist_km * 1000

    for i, row in st_utm.iterrows():
        if i in used:
            continue
        selected_idx.append(i)
        dists = st_utm.geometry.distance(row.geometry)
        used.update(dists[dists < min_dist_m].index)

    return g_st.loc[selected_idx].copy()
