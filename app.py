"""
NOAA GHCND Station Selector for SWAT+
======================================
Upload any watershed shapefile, download NOAA climate stations,
filter by quality, analyze climate, and export SWAT+ ready files.

Run with:  streamlit run app.py
"""
import datetime
import html
import io
import os
import tempfile
import zipfile

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import MarkerCluster
from scipy.stats import linregress
from streamlit_folium import st_folium
from scripts.download_noaa import download_noaa_stations
from scripts.filter_and_score import filter_and_score
from scripts.select_stations import select_stations
from scripts.analyze_subbasins import analyze_subbasins
from scripts.download_noaa_daily import download_noaa_daily
from scripts.download_noaa_yearly import download_noaa_yearly
from scripts.make_map import make_map
from scripts.noaa_single import get_station_daily_from_existing_workflow
from scripts.export_swatplus import export_swatplus
from scripts import session_store

# Daily records on NOAA's server start as early as the 1800s; allow the full range.
EARLIEST_DATE = datetime.date(1900, 1, 1)
TODAY         = datetime.date.today()

# Free basemaps (CARTO "positron" now answers every tile with "API KEY REQUIRED").
ESRI_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/{svc}/MapServer/tile/{{z}}/{{y}}/{{x}}"


# ══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL CACHED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_shapefile(shp_path):
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("ESRI:54003")
    return gdf.to_crs(4326)


def folder_sig(folder: str) -> tuple:
    """Name, size and modification time of every CSV in a folder.

    Passed to the cached loaders so they reload when a file is replaced
    (st.cache_data keyed only on the folder/path kept serving the old data).
    """
    if not os.path.isdir(folder):
        return ()
    out = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(".csv"):
            st_ = os.stat(os.path.join(folder, f))
            out.append((f, st_.st_size, st_.st_mtime_ns))
    return tuple(out)


@st.cache_data
def csv_span(path: str, mtime: float):
    """(first_date, last_date) of a station CSV, read from its first and last lines."""
    try:
        with open(path, "rb") as fh:
            fh.readline()                                   # header
            first = fh.readline().decode(errors="ignore").split(",")[0]
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 400))
            tail = fh.read().decode(errors="ignore").strip().splitlines()
        last = tail[-1].split(",")[0] if tail else ""
        return pd.to_datetime(first, errors="coerce"), pd.to_datetime(last, errors="coerce")
    except OSError:
        return pd.NaT, pd.NaT


@st.cache_data
def load_station_data(path, mtime=0.0):
    df = pd.read_csv(path)
    df["DATE"] = pd.to_datetime(df["DATE"])
    return df


@st.cache_data
def load_all_stations(noaa_folder: str, sig: tuple = ()) -> dict:
    """Load every CSV in noaa_folder into a dict keyed by station ID."""
    files = [f for f in os.listdir(noaa_folder) if f.endswith(".csv")]
    data = {}
    for f in files:
        sid = f.replace(".csv", "").replace("_", ":")
        df = pd.read_csv(os.path.join(noaa_folder, f))
        df["DATE"] = pd.to_datetime(df["DATE"])
        df["year"] = df["DATE"].dt.year
        data[sid] = df
    return data


@st.cache_data
def load_all_noaa_daily(folder: str, sig: tuple = ()) -> pd.DataFrame:
    """Concatenate all CSVs in folder into a single DataFrame."""
    dfs = []
    for f in os.listdir(folder):
        if f.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, f))
            df["DATE"] = pd.to_datetime(df["DATE"])
            df["year"] = df["DATE"].dt.year
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def compute_station_metrics(df: pd.DataFrame):
    """Return (mean_prcp, prcp_trend, missing_pct, tmax_trend, tmin_trend)."""
    df2 = df.copy()
    df2["year"] = df2["DATE"].dt.year
    annual = df2.groupby("year")["PRCP"].sum()
    mean_prcp = annual.mean()
    missing_pct = df2["PRCP"].isna().mean() * 100

    def _trend(series):
        series = series.dropna()
        if len(series) > 1:
            slope, *_ = linregress(series.index.astype(float), series.values)
            return slope * 10
        return np.nan

    prcp_trend = _trend(annual)
    tmax_trend = _trend(df2.groupby("year")["TMAX"].mean())
    tmin_trend = _trend(df2.groupby("year")["TMIN"].mean())
    return mean_prcp, prcp_trend, missing_pct, tmax_trend, tmin_trend


# ══════════════════════════════════════════════════════════════════════════════
# SMALL UI BADGE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _missing_badge(pct: float) -> str:
    if pct < 5:
        return f"🟢 {pct:.1f}% missing — Excellent"
    elif pct < 15:
        return f"🟡 {pct:.1f}% missing — Acceptable"
    else:
        return f"🔴 {pct:.1f}% missing — High (review before using in SWAT)"


def _coverage_badge(frac: float) -> str:
    if frac >= 0.8:
        return f"🟢 {frac:.0%} coverage — Good"
    elif frac >= 0.5:
        return f"🟡 {frac:.0%} coverage — Marginal"
    else:
        return f"🔴 {frac:.0%} coverage — Low (consider excluding)"


def _station_count_badge(n: int, area_km2: float) -> str:
    """
    WMO (2008) minimum gauge density guidelines:
      Interior plains : 575 km2/station
      Mountainous     : 250 km2/station
    SWAT literature shows satisfactory results with as few as 4 stations.
    We use the WMO plains threshold as the recommended minimum.
    """
    wmo_min  = max(4, int(area_km2 / 575))   # WMO plains standard
    swat_min = 4                               # SWAT practical minimum
    if n >= wmo_min:
        return (f"🟢 {n} stations — excellent coverage "
                f"(WMO plains minimum for {area_km2:,.0f} km² = {wmo_min} stations)")
    elif n >= swat_min:
        return (f"🟡 {n} stations — below WMO minimum ({wmo_min}) but above "
                f"SWAT practical minimum ({swat_min}). Acceptable for most applications.")
    else:
        return (f"🔴 Only {n} stations — below SWAT practical minimum of {swat_min}. "
                "Try relaxing filters in Step 2.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def run_app():
    st.set_page_config(
        page_title="NOAA Station Selector for SWAT+",
        page_icon="🌧️",
        layout="wide",
    )

    language = st.sidebar.selectbox("Language / Idioma", ["English", "Español"])

    def T(en, es):
        return en if language == "English" else es

    st.title(T(
        "🌧️ Climate Station Selector",
        "🌧️ Selector de Estaciones Climáticas"
    ))

    # ── Session recovery ───────────────────────────────────────────────────────
    # Streamlit starts from scratch whenever a session resets (reconnect, app restart,
    # page reload). The basin, parameters and station selection are therefore also kept
    # on the server under a random code in the URL (?s=...) and restored from it.
    # The NOAA token is never stored.
    sid = st.query_params.get("s")
    if not session_store.valid_token(sid):
        sid = session_store.new_token()
        st.query_params["s"] = sid
    st.session_state["_sid"] = sid

    if not st.session_state.get("_restore_tried"):
        st.session_state["_restore_tried"] = True
        session_store.purge_old()
        saved = session_store.load_all(sid) if "basin" not in st.session_state else {}
        if "basin" in saved:
            ss_ = st.session_state
            meta = saved.get("meta", {})
            ss_["basin"]      = saved["basin"]
            ss_["basin_name"] = meta.get("basin_name", "")
            ss_["area_km2"]   = meta.get("area_km2") or                 saved["basin"].to_crs("EPSG:6933").geometry.area.sum() / 1e6
            raw_sig = meta.get("basin_sig")
            ss_["basin_sig"]  = tuple(tuple(x) for x in raw_sig) if raw_sig else None
            if "params" in saved:
                ss_["params"] = {**saved["params"], "token": ""}
            if "stations" in saved and "buffer" in saved:
                ss_["stations_thinned"] = saved["stations"]
                ss_["basin_buffer"]     = saved["buffer"]
            st.toast(T("♻️ Restored your previous basin, parameters and results "
                       "(re-enter your NOAA token to run a new search).",
                       "♻️ Se recuperaron tu cuenca, parámetros y resultados anteriores "
                       "(vuelve a escribir tu token NOAA para hacer otra búsqueda)."))

    def _persist():
        """Write whatever changed since last time to the server (never breaks the app)."""
        ss_ = st.session_state
        sid_ = ss_.get("_sid")
        if not session_store.valid_token(sid_):
            return
        done = ss_.setdefault("_persisted", {})
        try:
            if "basin" not in ss_:
                if done:
                    session_store.delete_session(sid_)
                    done.clear()
                return
            sig = ss_.get("basin_sig") or ("n", len(ss_["basin"]))
            if done.get("basin") != sig:
                session_store.save_part(sid_, "basin", ss_["basin"])
                bs = ss_.get("basin_sig")
                session_store.save_part(sid_, "meta", {
                    "basin_name": ss_.get("basin_name", ""),
                    "area_km2": float(ss_.get("area_km2", 0) or 0),
                    "basin_sig": [list(x) for x in bs] if bs else None})
                done["basin"] = sig

            P_ = ss_.get("params")
            if P_ is not None:
                safe = {k: v for k, v in P_.items() if k != "token"}
                psig = repr(sorted(safe.items()))
                if done.get("params") != psig:
                    session_store.save_part(sid_, "params", safe)
                    done["params"] = psig

            st_ = ss_.get("stations_thinned")
            if st_ is not None and "basin_buffer" in ss_ and len(st_) and "id" in st_.columns:
                ssig = (len(st_), tuple(st_["id"]),
                        round(float(st_["score"].sum()), 4) if "score" in st_.columns else 0)
                if done.get("stations") != ssig:
                    session_store.save_part(sid_, "stations", st_)
                    session_store.save_part(sid_, "buffer", ss_["basin_buffer"])
                    done["stations"] = ssig
            elif done.get("stations") is not None:
                session_store.delete_part(sid_, "stations")
                session_store.delete_part(sid_, "buffer")
                done.pop("stations", None)
        except Exception:
            pass

    _persist()

    step = st.sidebar.radio(
        T("Navigation", "Navegación"),
        ["1️⃣ Upload", "2️⃣ Parameters", "3️⃣ Results", "4️⃣ Analysis", "5️⃣ Export SWAT+"]
    )

    # Streamlit drops a widget's value when the widget is not drawn in a run (i.e. when
    # you visit another step). Re-assigning the key keeps it, so folders and date
    # ranges survive moving between steps.
    for _k in ("noaa_folder_download", "dl_start", "dl_end", "exp_noaa_folder",
               "exp_start", "exp_end", "exp_output_folder"):
        if _k in st.session_state:
            st.session_state[_k] = st.session_state[_k]
    
    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — Upload shapefile
    # ══════════════════════════════════════════════════════════════════════════
    if step == "1️⃣ Upload":

        st.header(T("1. Upload your watershed shapefile",
                    "1. Cargar shapefile de la cuenca"))

        with st.expander(T("ℹ️ What is this step for?",
                           "ℹ️ ¿Para qué es este paso?"), expanded=True):
            st.markdown(T(
                """
                Your watershed boundary (shapefile) defines **where** SWAT will simulate hydrology.
                This app will automatically:
                1. Download all NOAA GHCND weather stations inside and around your watershed
                2. Score and filter them by data quality and coverage
                3. Help you choose the best **calibration period** for your SWAT model
                4. Show climate trends so you understand what your model will be learning

                > 💡 **Why does this matter?** SWAT is only as good as its climate inputs.
                > A station with 40% missing data or no overlap with your calibration period
                > will introduce errors that no parameter tuning can fix.
                """,
                """
                El límite de tu cuenca (shapefile) define **dónde** SWAT simulará la hidrología.
                Esta app automáticamente:
                1. Descarga todas las estaciones NOAA GHCND dentro y alrededor de tu cuenca
                2. Las puntúa y filtra por calidad y cobertura de datos
                3. Te ayuda a elegir el mejor **periodo de calibración** para SWAT
                4. Muestra tendencias climáticas para entender qué aprenderá el modelo

                > 💡 **¿Por qué importa?** SWAT es tan bueno como sus entradas climáticas.
                > Una estación con 40% de datos faltantes o sin coincidencia con tu periodo
                > de calibración introducirá errores que ningún ajuste de parámetros puede corregir.
                """
            ))

        # The uploader widget forgets its files as soon as you leave this step, but the
        # basin itself lives in session_state. Show what is loaded, and only re-read
        # the shapefile when the uploaded files actually change.
        uploaded_files = st.file_uploader(
            T("Upload all 4 shapefile components (.shp .shx .dbf .prj)",
              "Sube los 4 componentes del shapefile (.shp .shx .dbf .prj)")
            if "basin" not in st.session_state else
            T("Replace the loaded shapefile (upload all 4 components)",
              "Reemplazar el shapefile cargado (sube los 4 componentes)"),
            type=["shp", "shx", "dbf", "prj"],
            accept_multiple_files=True,
            key=f"shp_uploader_{st.session_state.get('uploader_gen', 0)}",
        )

        if uploaded_files:
            upload_sig = tuple(sorted((f.name, f.size) for f in uploaded_files))

            if st.session_state.get("basin_sig") != upload_sig:
                required_ext = {".shp", ".shx", ".dbf", ".prj"}
                uploaded_ext = {os.path.splitext(f.name)[1].lower() for f in uploaded_files}
                missing_ext = required_ext - uploaded_ext
                if missing_ext:
                    st.error(T(
                        f"Missing components: {missing_ext}. Please upload all four files.",
                        f"Faltan componentes: {missing_ext}. Sube los cuatro archivos."
                    ))
                    st.stop()

                temp_dir = tempfile.mkdtemp()
                for f in uploaded_files:
                    with open(os.path.join(temp_dir, f.name), "wb") as out:
                        out.write(f.getvalue())

                shp_path = next(
                    os.path.join(temp_dir, f.name)
                    for f in uploaded_files if f.name.lower().endswith(".shp")
                )
                basin = load_shapefile(shp_path)

                if basin.crs is None:
                    st.warning(T(
                        "No CRS detected. Assigning Miller Cylindrical (ESRI:54003).",
                        "Sin CRS detectado. Asignando Miller Cylindrical (ESRI:54003)."
                    ))
                    basin = basin.set_crs("ESRI:54003")
                basin = basin.to_crs(4326)

                # A different basin invalidates everything derived from the old one.
                for key in ["stations_thinned", "basin_buffer", "final_map", "run",
                            "results_map", "results_map_sig"]:
                    st.session_state.pop(key, None)

                st.session_state["basin"]      = basin
                st.session_state["basin_sig"]  = upload_sig
                st.session_state["basin_name"] = os.path.basename(shp_path)
                st.session_state["area_km2"]   = basin.to_crs("EPSG:6933").geometry.area.sum() / 1e6  # equal-area, any location

        _persist()

        # ── Loaded basin (stays visible while you change parameters) ───────────
        if "basin" in st.session_state:
            basin    = st.session_state["basin"]
            area_km2 = st.session_state["area_km2"]
            n_subs   = len(basin)
            bounds   = basin.total_bounds
            rec_stations = max(4, int(area_km2 / 575))  # WMO plains standard

            col_ok, col_clear = st.columns([5, 1])
            col_ok.success(T(
                f"✅ Shapefile loaded: {st.session_state.get('basin_name', '')}",
                f"✅ Shapefile cargado: {st.session_state.get('basin_name', '')}"))
            if col_clear.button(T("🗑️ Remove", "🗑️ Quitar"), use_container_width=True):
                for key in ["basin", "basin_sig", "basin_name", "area_km2", "stations_thinned",
                            "basin_buffer", "final_map", "run", "results_map", "results_map_sig"]:
                    st.session_state.pop(key, None)
                st.session_state["uploader_gen"] = st.session_state.get("uploader_gen", 0) + 1
                _persist()      # basin is gone -> deletes the saved copy
                st.rerun()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric(T("Total area", "Área total"), f"{area_km2:,.0f} km²")
            col2.metric(T("Subbasins", "Subcuencas"), n_subs)
            col3.metric(T("Recommended stations", "Estaciones recomendadas"), f"≥ {rec_stations}")
            col4.metric(T("Lon range", "Rango Lon"),
                        f"{bounds[0]:.1f}° – {bounds[2]:.1f}°")

            with st.expander(T("📐 What do these numbers mean?",
                               "📐 ¿Qué significan estos números?")):
                st.markdown(T(
                    f"""
                    - **{area_km2:,.0f} km²** — Total watershed area.
                      The **WMO (2008)** recommends 1 station per **575 km²** for interior plains
                      and 1 per **250 km²** for mountains. For SWAT specifically, research shows
                      that as few as **4 well-distributed stations** can yield satisfactory results —
                      more stations help mainly in large or topographically complex basins.
                    - **{n_subs} subbasins** — Each subbasin in SWAT gets assigned its nearest
                      weather station. More subbasins = more spatial resolution needed.
                    - **≥ {rec_stations} stations recommended** — WMO plains minimum for your basin.
                      Beyond this threshold, adding more stations yields diminishing returns.
                    """,
                    f"""
                    - **{area_km2:,.0f} km²** — Área total de la cuenca.
                      La **OMM (2008)** recomienda 1 estación por **575 km²** en planicies
                      y 1 por **250 km²** en zonas montañosas. Para SWAT, la literatura muestra
                      que con tan solo **4 estaciones bien distribuidas** se pueden obtener
                      resultados satisfactorios — más estaciones ayudan principalmente en cuencas
                      grandes o topográficamente complejas.
                    - **{n_subs} subcuencas** — Cada subcuenca en SWAT usa la estación más cercana.
                      Más subcuencas = más resolución espacial necesaria.
                    - **≥ {rec_stations} estaciones recomendadas** — Mínimo OMM para tu cuenca.
                      Más allá de este umbral los rendimientos son decrecientes.
                    """
                ))

            st.info(T(
                "✅ Basin loaded. Go to **2️⃣ Parameters** to configure the station search.",
                "✅ Cuenca cargada. Ve a **2️⃣ Parameters** para configurar la búsqueda."
            ))

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — Parameters
    # ══════════════════════════════════════════════════════════════════════════
    elif step == "2️⃣ Parameters":

        st.header(T("2. Configure station search",
                    "2. Configurar búsqueda de estaciones"))

        with st.expander(T("📖 How to choose these parameters",
                           "📖 Cómo elegir estos parámetros"), expanded=True):
            st.markdown(T(
                """
                | Parameter | What it does | Guidance |
                |---|---|---|
                | **Buffer (km)** | Expands search area beyond watershed | 15–30 km for mountains (sparse stations), 5–10 km for flat data-rich regions |
                | **Minimum year** | Excludes stations inactive before this year | Set to your calibration start year |
                | **Calibration period** | The window SWAT will calibrate on | Must include **both wet and dry years** — calibrating only on droughts or floods biases the model |
                | **Min years of data** | Minimum total record length | At least 5 years; 10+ is ideal for reliable parameter estimation |
                | **Coverage fraction** | How much of calibration window the station covers | 0.7 = must cover 70% of your window; below 0.5 introduces systematic bias |
                | **Min spacing (km)** | Min distance between selected stations | 25 km prevents redundancy — two stations 5 km apart add no new spatial info to SWAT |
                """,
                """
                | Parámetro | Qué hace | Guía |
                |---|---|---|
                | **Buffer (km)** | Amplía el área de búsqueda más allá de la cuenca | 15–30 km en cuencas montañosas, 5–10 km en regiones planas |
                | **Año mínimo** | Excluye estaciones inactivas antes de este año | Ponlo igual al inicio de tu calibración |
                | **Periodo de calibración** | Ventana que SWAT usará para calibrarse | Debe incluir **años secos Y húmedos** — calibrar solo en sequía o solo en inundaciones sesga el modelo |
                | **Años mínimos** | Longitud mínima del registro | Mínimo 5 años; 10+ es ideal para estimación confiable de parámetros |
                | **Fracción de cobertura** | Qué tanto del periodo cubre la estación | 0.7 = debe cubrir el 70%; menos de 0.5 introduce sesgo sistemático |
                | **Distancia mínima (km)** | Distancia mínima entre estaciones seleccionadas | 25 km evita redundancia — dos estaciones a 5 km no aportan información espacial nueva |
                """
            ))

        # Widgets forget their value when you leave this step, so seed each one with
        # what was saved last time (st.session_state["params"] is rewritten below).
        P = st.session_state.get("params", {})

        if "basin" in st.session_state:
            st.caption(T(f"📍 Basin loaded: {st.session_state.get('basin_name', '')}",
                         f"📍 Cuenca cargada: {st.session_state.get('basin_name', '')}"))
        else:
            st.warning(T("No shapefile loaded yet — go to **1️⃣ Upload** first.",
                         "Aún no hay shapefile — ve primero a **1️⃣ Upload**."))

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(T("**Search area**", "**Área de búsqueda**"))
            buffer_km = st.number_input(
                T("Buffer (km)", "Buffer (km)"), 0, 100, int(P.get("buffer_km", 15)),
                help=T("Expand search area beyond watershed boundary",
                       "Ampliar el área de búsqueda más allá de la cuenca")
            )
            min_year = st.number_input(
                T("Minimum active year", "Año mínimo activo"), 1900, TODAY.year, int(P.get("min_year", 1900)),
                help=T("Exclude stations that stopped recording before this year. "
                       "1900 keeps every station whose record overlaps your window.",
                       "Excluir estaciones que dejaron de registrar antes de este año. "
                       "1900 conserva toda estación cuyo registro se solape con tu ventana.")
            )

        with col2:
            st.markdown(T("**Calibration window**", "**Ventana de calibración**"))
            cal_start = st.date_input(
                T("Calibration start", "Inicio de calibración"),
                value=P.get("cal_start", datetime.date(2010, 1, 1)),
                min_value=datetime.date(1900, 1, 1),
                max_value=datetime.date(2100, 12, 31),
            )
            cal_end = st.date_input(
                T("Calibration end", "Fin de calibración"),
                value=P.get("cal_end", datetime.date(2019, 12, 31)),
                min_value=datetime.date(1900, 1, 1),
                max_value=datetime.date(2100, 12, 31),
            )
            cal_len = (pd.Timestamp(cal_end) - pd.Timestamp(cal_start)).days / 365.25
            if cal_len < 5:
                st.warning(T(
                    f"⚠️ {cal_len:.1f} years — very short for SWAT calibration. Aim for ≥ 10 years.",
                    f"⚠️ {cal_len:.1f} años — muy corto. Apunta a ≥ 10 años."
                ))
            elif cal_len < 10:
                st.info(T(
                    f"ℹ️ {cal_len:.1f} years — acceptable, but 10+ is preferred.",
                    f"ℹ️ {cal_len:.1f} años — aceptable, pero se prefieren 10+."
                ))
            else:
                st.success(T(
                    f"✅ {cal_len:.1f} years — good calibration window.",
                    f"✅ {cal_len:.1f} años — buena ventana de calibración."
                ))

        with col3:
            st.markdown(T("**Quality filters**", "**Filtros de calidad**"))
            min_years = st.number_input(
                T("Min years of data", "Años mínimos de datos"), 1, 130, int(P.get("min_years", 5)),
                help=T("Minimum total record length",
                       "Longitud mínima total del registro")
            )
            min_cover = st.slider(
                T("Coverage fraction", "Fracción de cobertura"), 0.0, 1.0, float(P.get("min_cover", 0.7)),
                help=T("Fraction of calibration period the station must cover",
                       "Fracción del periodo de calibración que debe cubrir la estación")
            )
            min_dist_km = st.number_input(
                T("Min station spacing (km)", "Distancia mínima entre estaciones (km)"),
                1, 100, int(P.get("min_dist_km", 25)),
                help=T("Prevents redundant nearby stations",
                       "Evita estaciones redundantes cercanas")
            )

        st.divider()

        st.markdown("🔑 **Get your free NOAA API token here:**")
        st.link_button(
            "Open NOAA Token Page",
            "https://www.ncdc.noaa.gov/cdo-web/token"
        )

        token = st.text_input(
        T("🔑 NOAA API token", "🔑 Token NOAA API"),
        type="password",
        value=P.get("token", ""),
        )
        
        st.session_state["params"] = {
            "buffer_km": buffer_km, "min_year": min_year,
            "cal_start": cal_start, "cal_end": cal_end,
            "min_years": min_years, "min_cover": min_cover,
            "min_dist_km": min_dist_km, "token": token,
        }
        _persist()

        run_button = st.button(
            T("🚀 Run NOAA Station Selection", "🚀 Ejecutar selección de estaciones NOAA"),
            type="primary"
        )

        if run_button:
            if "basin" not in st.session_state:
                st.error(T("Upload a shapefile in Step 1 first.",
                           "Sube un shapefile en el Paso 1 primero."))
                st.stop()
            if not token:
                st.error(T("A NOAA API token is required.",
                           "Se requiere un token NOAA API."))
                st.stop()
            for key in ["stations_thinned", "basin_buffer"]:
                st.session_state.pop(key, None)
            st.session_state["run"] = True
            st.success(T(
                "✅ Parameters saved. Navigate to **3️⃣ Results**.",
                "✅ Parámetros guardados. Ve a **3️⃣ Results**."
            ))

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — Results
    # ══════════════════════════════════════════════════════════════════════════
    elif step == "3️⃣ Results":

        if st.session_state.get("run"):
            if "basin" not in st.session_state or "params" not in st.session_state:
                st.error(T("Complete Steps 1 and 2 first.",
                           "Completa los Pasos 1 y 2 primero."))
                st.stop()

            basin  = st.session_state["basin"]
            params = st.session_state["params"]

            with st.spinner(T("Downloading and filtering NOAA stations…",
                               "Descargando y filtrando estaciones NOAA…")):
                try:
                    stations_thinned, basin_buffer = select_stations(
                        basin,
                        params["buffer_km"], params["min_year"],
                        params["cal_start"], params["cal_end"],
                        params["min_years"], params["min_cover"],
                        params["min_dist_km"], params["token"],
                    )
                except Exception as exc:
                    st.session_state["run"] = False
                    st.error(T(f"❌ The NOAA station search failed: {exc}",
                               f"❌ Falló la búsqueda de estaciones NOAA: {exc}"))
                    st.stop()

            st.session_state["stations_thinned"] = stations_thinned
            st.session_state["basin_buffer"]     = basin_buffer
            st.session_state["run"] = False
            _persist()

        if not all(k in st.session_state for k in ["stations_thinned", "basin_buffer"]):
            st.info(T(
                "No results yet. Configure parameters in Step 2 and click Run.",
                "Sin resultados aún. Configura parámetros en el Paso 2 y haz clic en Ejecutar."
            ))
            st.stop()

        basin            = st.session_state["basin"]
        stations_thinned = st.session_state["stations_thinned"]
        basin_buffer     = st.session_state["basin_buffer"]
        area_km2         = st.session_state.get("area_km2", 0)
        params           = st.session_state.get("params", {})

        st.header(T("3. Results", "3. Resultados"))

        if stations_thinned.empty:
            st.error(T(
                "❌ No stations found. Try increasing the buffer, lowering coverage fraction, "
                "or reducing min years.",
                "❌ No se encontraron estaciones. Intenta aumentar el buffer, reducir la "
                "fracción de cobertura o los años mínimos."
            ))
            st.stop()

        # ── Quality semaphore ──────────────────────────────────────────────
        st.subheader(T("📊 Selection quality report",
                       "📊 Reporte de calidad de la selección"))

        n_stations = len(stations_thinned)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(T("Stations selected", "Estaciones seleccionadas"), n_stations)
        if "score" in stations_thinned.columns:
            col2.metric(T("Avg score", "Score promedio"),
                        f"{stations_thinned['score'].mean():.2f}")
        if "cover_frac" in stations_thinned.columns:
            col3.metric(T("Avg coverage", "Cobertura promedio"),
                        f"{stations_thinned['cover_frac'].mean():.0%}")
        if "span_years" in stations_thinned.columns:
            col4.metric(T("Avg record length", "Longitud promedio"),
                        f"{stations_thinned['span_years'].mean():.0f} yrs")

        # Station count badge
        st.markdown(_station_count_badge(n_stations, area_km2))

        # Coverage quality warnings
        if "cover_frac" in stations_thinned.columns:
            low_cov = (stations_thinned["cover_frac"] < 0.6).sum()
            if low_cov > 0:
                st.warning(T(
                    f"⚠️ {low_cov} station(s) cover < 60% of your calibration period. "
                    "Consider excluding them or narrowing your calibration window.",
                    f"⚠️ {low_cov} estación(es) cubren < 60% del periodo de calibración. "
                    "Considera excluirlas o reducir tu ventana de calibración."
                ))

        # Elevation band coverage
        if "elev_band" in stations_thinned.columns:
            bands = stations_thinned["elev_band"].value_counts()
            missing_bands = {"baja", "media-baja", "media-alta", "alta"} - set(bands.index)
            if missing_bands:
                st.warning(T(
                    f"⚠️ No stations in elevation band(s): {missing_bands}. "
                    "SWAT may poorly represent precipitation at those elevations.",
                    f"⚠️ Sin estaciones en banda(s) de elevación: {missing_bands}. "
                    "SWAT puede representar mal la precipitación en esas elevaciones."
                ))
            else:
                st.success(T("✅ All elevation bands covered.",
                             "✅ Todas las bandas de elevación están cubiertas."))

        # Station table + download
        with st.expander(T("📋 Station details table",
                           "📋 Tabla detallada de estaciones")):
            display_cols = [c for c in ["name", "id", "elevation", "elev_band",
                                         "span_years", "cover_frac", "score",
                                         "mindate", "maxdate"]
                            if c in stations_thinned.columns]
            st.dataframe(stations_thinned[display_cols].reset_index(drop=True),
                         use_container_width=True)
            csv_bytes = stations_thinned[display_cols].to_csv(index=False).encode()
            st.download_button(
                T("⬇️ Download station list (CSV)",
                  "⬇️ Descargar lista de estaciones (CSV)"),
                data=csv_bytes,
                file_name="selected_stations.csv",
                mime="text/csv"
            )

        # ── Folium map ─────────────────────────────────────────────────────
        st.subheader(T("🗺️ Station map", "🗺️ Mapa de estaciones"))

        with st.expander(T("ℹ️ How to read this map",
                           "ℹ️ Cómo leer este mapa")):
            st.markdown(T(
                """
                - **Marker colors** indicate elevation band:
                  🔵 Low · 🟢 Mid-low · 🟠 Mid-high · 🔴 High
                - **Dashed orange polygon**: buffer zone used to search for stations
                - **Blue polygon**: your watershed boundary
                - **Gray lines**: individual subbasins — each gets its nearest station in SWAT
                - Click any marker to see station name, coverage, and score
                """,
                """
                - **Colores de marcadores** indican banda de elevación:
                  🔵 Baja · 🟢 Media-baja · 🟠 Media-alta · 🔴 Alta
                - **Polígono naranja punteado**: zona buffer usada para buscar estaciones
                - **Polígono azul**: límite de tu cuenca
                - **Líneas grises**: subcuencas individuales — cada una usa la estación más cercana en SWAT
                - Haz clic en cualquier marcador para ver nombre, cobertura y score
                """
            ))

        # Build the folium map once per selection and keep it: rebuilding it (and
        # re-sending ~3 MB of GeoJSON) on every Streamlit rerun was the slow part.
        map_sig = (language, tuple(stations_thinned["id"]), len(basin),
                   tuple(np.round(basin.total_bounds, 5)),
                   tuple(np.round(basin_buffer.total_bounds, 5)))
        if st.session_state.get("results_map_sig") != map_sig:
            stations     = stations_thinned.copy()
            # Draw geometry only: uploaded shapefiles often carry date/time or other
            # attribute columns that cannot be written to GeoJSON, which used to make
            # the map crash for basins other than the bundled one.
            basin_subs   = gpd.GeoDataFrame(
                geometry=basin.geometry.simplify(0.002).set_precision(1e-4), crs=4326)  # ~10 m: halves the GeoJSON
            basin_union  = basin.union_all()
            basin_single = gpd.GeoDataFrame(geometry=[basin_union], crs=4326)
            basin_buffer_geo = gpd.GeoDataFrame(geometry=basin_buffer.geometry, crs=4326)

            def color_from_band(band):
                return {"baja": "blue", "media-baja": "green",
                        "media-alta": "orange", "alta": "red"}.get(band, "gray")

            centroid = basin_union.centroid
            m_final  = folium.Map(location=[centroid.y, centroid.x],
                                  zoom_start=7, tiles=None)
            folium.TileLayer(tiles="OpenStreetMap", name="OpenStreetMap").add_to(m_final)
            folium.TileLayer(
                tiles=ESRI_TILES.format(svc="Canvas/World_Light_Gray_Base"),
                attr="Tiles &copy; Esri", name="Light gray (Esri)",
                max_zoom=16, show=False).add_to(m_final)
            folium.TileLayer(
                tiles=ESRI_TILES.format(svc="World_Imagery"),
                attr="Tiles &copy; Esri, Maxar, Earthstar Geographics",
                name="Satellite (Esri)", max_zoom=18, show=False).add_to(m_final)
            bw, bs, be, bn = basin_buffer_geo.total_bounds
            m_final.fit_bounds([[bs, bw], [bn, be]])

            folium.GeoJson(
                basin_single.to_json(), name=T("Basin", "Cuenca"),
                style_function=lambda _: {"fillColor": "#3388ff", "fillOpacity": 0.05,
                                           "color": "#003399", "weight": 2}
            ).add_to(m_final)
            folium.GeoJson(
                basin_subs.to_json(), name=T("Subbasins", "Subcuencas"),
                style_function=lambda _: {"fillOpacity": 0, "color": "#666", "weight": 1}
            ).add_to(m_final)
            folium.GeoJson(
                basin_buffer_geo.to_json(), name=T("Buffer", "Buffer"),
                style_function=lambda _: {"fillOpacity": 0, "color": "#f90",
                                           "weight": 2, "dashArray": "6 4"}
            ).add_to(m_final)

            cluster = MarkerCluster(name=T("Stations", "Estaciones")).add_to(m_final)
            for row in stations.itertuples():
                band = getattr(row, "elev_band", "NA")
                cov  = getattr(row, "cover_frac", float("nan"))
                scr  = getattr(row, "score", float("nan"))
                popup_html = (
                    f"<b>{html.escape(str(row.name))}</b><br>"
                    f"Elevation band: {band}<br>"
                    f"Coverage: {cov:.0%}<br>"
                    f"Score: {scr:.3f}"
                )
                folium.Marker(
                    location=[row.geometry.y, row.geometry.x],
                    popup=folium.Popup(popup_html, max_width=250),
                    icon=folium.Icon(color=color_from_band(band), icon="cloud"),
                ).add_to(cluster)

            folium.LayerControl().add_to(m_final)
            st.session_state["results_map"]     = m_final
            st.session_state["results_map_sig"] = map_sig
        m_final = st.session_state["results_map"]
        # returned_objects=[]: panning/zooming must NOT rerun the whole page.
        st_folium(m_final, width=900, height=600, returned_objects=[],
                  key="results_map_widget")

        # ── Narrative handoff ──────────────────────────────────────────────
        cal_s    = params.get("cal_start", "?")
        cal_e    = params.get("cal_end", "?")
        avg_cov  = (stations_thinned["cover_frac"].mean()
                    if "cover_frac" in stations_thinned.columns else float("nan"))
        avg_scr  = (stations_thinned["score"].mean()
                    if "score" in stations_thinned.columns else float("nan"))

        st.info(T(
            f"**Summary:** {n_stations} stations · calibration {cal_s} → {cal_e} · "
            f"avg coverage {avg_cov:.0%} · avg score {avg_scr:.2f}  \n"
            "➡️ Go to **4️⃣ Analysis** to inspect data quality, trends, and find the "
            "optimal calibration period.",
            f"**Resumen:** {n_stations} estaciones · calibración {cal_s} → {cal_e} · "
            f"cobertura promedio {avg_cov:.0%} · score promedio {avg_scr:.2f}  \n"
            "➡️ Ve a **4️⃣ Analysis** para inspeccionar calidad de datos, tendencias y "
            "encontrar el periodo óptimo."
        ))

        # ── Daily data download panel ──────────────────────────────────────
        st.divider()
        st.subheader(T("📥 Daily Data Status & Download",
                       "📥 Estado y Descarga de Datos Diarios"))

        with st.expander(T("ℹ️ What is this section for?",
                           "ℹ️ ¿Para qué es esta sección?"), expanded=True):
            st.markdown(T(
                """
                The **Analysis tab (Step 4)** reads daily CSV files from a local folder
                (`noaa_raw/` by default). Each file contains the full daily record
                (PRCP, TMAX, TMIN) for one station.

                This section shows which of your selected stations already have data
                downloaded, and lets you download the missing ones directly from the
                NOAA CDO API — using the token you entered in Step 2.

                > 💡 **Already have data?** If you previously downloaded 66 station files,
                > they will be detected automatically. You only need to download stations
                > that appear as ❌ Missing below.
                """,
                """
                El **tab de Análisis (Paso 4)** lee archivos CSV diarios de una carpeta
                local (`noaa_raw/` por defecto). Cada archivo contiene el registro diario
                completo (PRCP, TMAX, TMIN) de una estación.

                Esta sección muestra qué estaciones seleccionadas ya tienen datos descargados
                y te permite descargar las faltantes directamente desde la API CDO de NOAA,
                usando el token que ingresaste en el Paso 2.

                > 💡 **¿Ya tienes datos?** Si previamente descargaste 66 archivos de
                > estaciones, se detectarán automáticamente. Solo necesitas descargar las
                > que aparezcan como ❌ Faltante abajo.
                """
            ))

        noaa_folder = st.text_input(
            T("📂 Data folder (where CSVs are / will be saved)",
              "📂 Carpeta de datos (donde están / se guardarán los CSVs)"),
            value="noaa_raw",
            key="noaa_folder_download"
        )
        st.session_state["noaa_folder_6b"] = noaa_folder

        # Scan which stations already have a CSV
        os.makedirs(noaa_folder, exist_ok=True)
        existing_files = {f.replace(".csv", "").replace("_", ":")
                          for f in os.listdir(noaa_folder) if f.endswith(".csv")}
        station_ids    = stations_thinned["id"].tolist() if "id" in stations_thinned.columns else []

        # Requested date range (chosen before the table so we can tell which files are cut short)
        # NB: without min_value Streamlit only lets you go back 10 years from
        # the default value (which is why this used to stop at 2000).
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            dl_start = st.date_input(
                T("Download from", "Descargar desde"),
                value=EARLIEST_DATE,
                min_value=EARLIEST_DATE, max_value=TODAY,
                key="dl_start",
                help=T("Full station record: data are fetched from NOAA's GHCN-Daily "
                       "files, which go back as far as each station reports (often before 1950).",
                       "Registro completo de la estación: los datos vienen de los archivos "
                       "GHCN-Daily de NOAA, que llegan hasta donde cada estación reporta "
                       "(a menudo antes de 1950).")
            )
        with col_dl2:
            dl_end = st.date_input(
                T("Download to", "Descargar hasta"),
                value=TODAY,
                min_value=EARLIEST_DATE, max_value=TODAY,
                key="dl_end"
            )

        # Inspect each existing file: a CSV that starts years after the station's first
        # record (e.g. downloaded earlier with a 2010 start) is "partial" and is
        # fetched again, instead of silently feeding 2010-only data to every figure.
        dl_done = st.session_state.setdefault("dl_done", set())
        info    = {}
        for sid_dl in station_ids:
            fpath_dl = os.path.join(noaa_folder, sid_dl.replace(":", "_") + ".csv")
            if not os.path.exists(fpath_dl):
                info[sid_dl] = {"state": "missing"}
                continue
            first, last = csv_span(fpath_dl, os.path.getmtime(fpath_dl))
            row_st = stations_thinned[stations_thinned["id"] == sid_dl].iloc[0]
            station_first = pd.to_datetime(row_st.get("mindate"), errors="coerce")
            wanted_from   = max(pd.Timestamp(dl_start),
                                station_first if pd.notna(station_first) else pd.Timestamp(dl_start))
            partial = (pd.notna(first) and (first.year - wanted_from.year) > 1
                       and (sid_dl, str(dl_start)) not in dl_done)
            info[sid_dl] = {"state": "partial" if partial else "ok", "first": first, "last": last,
                            "size_kb": os.path.getsize(fpath_dl) / 1024}

        already  = [s for s in station_ids if info[s]["state"] in ("ok", "partial")]
        missing  = [s for s in station_ids if info[s]["state"] == "missing"]
        partial  = [s for s in station_ids if info[s]["state"] == "partial"]
        to_get   = missing + partial
        extra    = [s for s in existing_files if s not in station_ids]

        # Status metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(T("✅ Downloaded",  "✅ Descargadas"),  len(already) - len(partial))
        col2.metric(T("⚠️ Start too late", "⚠️ Empiezan tarde"), len(partial),
                    help=T("Files that start years after the station's first record; they are downloaded again",
                           "Archivos que empiezan años después del primer registro de la estación; se vuelven a descargar"))
        col3.metric(T("⬇️ Downloadable",     "⬇️ Disponibles para descargar"),    len(missing))
        col4.metric(T("📦 Extra files", "📦 Archivos extra"), len(extra),
                    help=T("Files in folder that are not in the current selection",
                           "Archivos en la carpeta que no están en la selección actual"))

        # Density context for the current download
        area_km2_dl = st.session_state.get("area_km2", 0)
        total_dl    = len(station_ids)
        if area_km2_dl > 0 and total_dl > 0:
            density_km2 = area_km2_dl / total_dl
            st.caption(T(
                f"Current density: **1 station per {density_km2:,.0f} km²** "
                f"for a {area_km2_dl:,.0f} km² basin.  "
                "WMO recommends 575 km²/station (plains) · 250 km²/station (mountains). "
                "SWAT literature: ≥ 4 stations yields satisfactory results.",
                f"Densidad actual: **1 estación por {density_km2:,.0f} km²** "
                f"para una cuenca de {area_km2_dl:,.0f} km².  "
                "OMM recomienda 575 km²/estación (planicies) · 250 km²/estación (montañas). "
                "Literatura SWAT: ≥ 4 estaciones da resultados satisfactorios."
            ))

        # Station status table
        if station_ids:
            status_rows = []
            for sid_dl in station_ids:
                name_dl = ""
                if "name" in stations_thinned.columns:
                    match_dl = stations_thinned[stations_thinned["id"] == sid_dl]
                    if not match_dl.empty:
                        name_dl = match_dl.iloc[0]["name"]
                inf = info[sid_dl]
                label = {"ok":      T("✅ Downloaded", "✅ Descargada"),
                         "partial": T("⚠️ Starts late", "⚠️ Empieza tarde"),
                         "missing": T("⬇️ Downloadable", "⬇️ Disponible para descargar")}[inf["state"]]
                has = inf["state"] != "missing"
                status_rows.append({
                    T("Station", "Estación"):  name_dl,
                    "ID":                      sid_dl,
                    T("Status", "Estado"):     label,
                    T("Data from", "Datos desde"): inf["first"].date().isoformat() if has and pd.notna(inf["first"]) else "—",
                    T("Data to", "Datos hasta"):   inf["last"].date().isoformat() if has and pd.notna(inf["last"]) else "—",
                    T("Size (KB)", "Tamaño (KB)"): f"{inf['size_kb']:.0f}" if has else "—",
                })
            st.dataframe(pd.DataFrame(status_rows), use_container_width=True)

        # The CSVs live in `noaa_folder` on the machine that runs the app. When the app is
        # hosted (Streamlit Cloud) that is the server, not this computer, so offer them
        # as a ZIP that the browser can save to the Downloads folder.
        if already:
            zip_files = [(sid_z, os.path.join(noaa_folder, sid_z.replace(":", "_") + ".csv"))
                         for sid_z in already]
            zip_sig = tuple((p_, os.path.getmtime(p_)) for _, p_ in zip_files)
            zc1, zc2 = st.columns([1, 1])
            if zc1.button(T(f"📦 Prepare ZIP of {len(zip_files)} station file(s)",
                            f"📦 Preparar ZIP de {len(zip_files)} archivo(s) de estaciones")):
                with st.spinner(T("Zipping station files…", "Comprimiendo archivos de estaciones…")):
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
                        for _, path_z in zip_files:
                            zf.write(path_z, arcname=os.path.basename(path_z))
                        list_cols = [c for c in ["name", "id", "elevation", "elev_band", "span_years",
                                                 "cover_frac", "score", "mindate", "maxdate"]
                                     if c in stations_thinned.columns]
                        zf.writestr("selected_stations.csv",
                                    stations_thinned[list_cols].to_csv(index=False))
                st.session_state["zip_data"] = (zip_sig, buf.getvalue())
            zip_data = st.session_state.get("zip_data")
            if zip_data and zip_data[0] == zip_sig:
                zc2.download_button(
                    T(f"⬇️ Download station data (ZIP, {len(zip_data[1]) / 1e6:.1f} MB)",
                      f"⬇️ Descargar datos de estaciones (ZIP, {len(zip_data[1]) / 1e6:.1f} MB)"),
                    data=zip_data[1],
                    file_name=f"noaa_station_data_{len(zip_files)}_stations.zip",
                    mime="application/zip",
                    type="primary")
            st.caption(T(
                f"Files are saved on the computer that runs the app, in `{noaa_folder}/`. "
                "On the hosted app that is the server, so use this ZIP to get a copy.",
                f"Los archivos se guardan en el equipo que ejecuta la app, en `{noaa_folder}/`. "
                "En la app en línea es el servidor, así que usa este ZIP para tener una copia."))

        # Download missing (and cut-short) stations
        if to_get:
            st.warning(T(
                f"{len(missing)} station(s) have no local data file"
                + (f" and {len(partial)} start years after their first record (they will be replaced)"
                   if partial else "")
                + ". Download them below to use the Analysis tab.",
                f"{len(missing)} estación(es) no tienen archivo local"
                + (f" y {len(partial)} empiezan años después de su primer registro (se reemplazarán)"
                   if partial else "")
                + ". Descárgalas abajo para usar el tab de Análisis."
            ))

            dl_token = params.get("token", "")

            if st.button(T(f"⬇️ Download {len(to_get)} station(s)",
                           f"⬇️ Descargar {len(to_get)} estación(es)"),
                         type="primary"):
                # no token needed: GHCN-Daily files are public (the token is only a fallback)
                progress_bar = st.progress(0)
                status_text  = st.empty()
                errors       = []

                for i, sid_dl in enumerate(to_get):
                    status_text.text(T(
                        f"Downloading {i+1}/{len(to_get)}: {sid_dl}…",
                        f"Descargando {i+1}/{len(to_get)}: {sid_dl}…"
                    ))
                    try:
                        from scripts.download_noaa_daily import download_noaa_daily
                        df_dl = download_noaa_daily(
                            sid_dl, dl_start, dl_end, token=dl_token
                        )
                        if df_dl.empty:
                            errors.append(T(
                                f"⚠️ {sid_dl} — no data in this date range",
                                f"⚠️ {sid_dl} — sin datos en este rango de fechas"
                            ))
                        else:
                            fname_save = sid_dl.replace(":", "_") + ".csv"
                            df_dl.to_csv(os.path.join(noaa_folder, fname_save), index=False)
                            dl_done.add((sid_dl, str(dl_start)))
                    except Exception as e:
                        errors.append(f"❌ {sid_dl}: {e}")

                    progress_bar.progress((i + 1) / len(to_get))

                status_text.empty()

                if errors:
                    st.warning(T("Some stations had issues:", "Algunas estaciones tuvieron problemas:"))
                    for err in errors:
                        st.write(err)

                n_downloaded = len(to_get) - len(errors)
                if n_downloaded > 0:
                    st.success(T(
                        f"✅ {n_downloaded} station(s) downloaded successfully to `{noaa_folder}/`.",
                        f"✅ {n_downloaded} estación(es) descargadas exitosamente en `{noaa_folder}/`."
                    ))
                    st.rerun()
        else:
            st.success(T(
                f"✅ All {len(already)} selected stations have local data files in `{noaa_folder}/`. "
                "Ready for Analysis.",
                f"✅ Las {len(already)} estaciones seleccionadas tienen archivos locales en `{noaa_folder}/`. "
                "Listo para el Análisis."
            ))

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4 — Analysis
    # ══════════════════════════════════════════════════════════════════════════
    elif step == "4️⃣ Analysis":

        if "stations_thinned" not in st.session_state:
            st.warning(T("Run Steps 1–3 first.",
                         "Ejecuta los Pasos 1–3 primero."))
            st.stop()

        st.header(T("4. Climate Analysis Dashboard",
                    "4. Dashboard de Análisis Climático"))

        stations = st.session_state["stations_thinned"]

        station_choice = st.selectbox(
            T("Select a station", "Selecciona una estación"),
            stations["name"] + " — " + stations["id"],
        )

        sid   = station_choice.split(" — ")[-1]
        fname = sid.replace(":", "_") + ".csv"
        path  = os.path.join(st.session_state.get("noaa_folder_6b", "noaa_raw"), fname)

        if not os.path.exists(path):
            st.error(T(f"Data file not found: {path}",
                       f"Archivo no encontrado: {path}"))
            st.stop()

        df  = load_station_data(path, os.path.getmtime(path))
        df2 = df.copy()
        df2["year"]   = df2["DATE"].dt.year
        df2["month"]  = df2["DATE"].dt.month
        df2["doy"]    = df2["DATE"].dt.dayofyear
        df2["decade"] = (df2["year"] // 10) * 10

        tab1, tab2, tab3, tab4 = st.tabs([
            T("📍 Station Overview",    "📍 Vista General"),
            T("📈 Time Series & Gaps",  "📈 Series de Tiempo y Vacíos"),
            T("🎯 Calibration Period",  "🎯 Periodo de Calibración"),
            T("🌡️ Climate Trends",     "🌡️ Tendencias Climáticas"),
        ])

        # ── TAB 1 — STATION OVERVIEW ──────────────────────────────────────
        with tab1:
            row = stations[stations["id"] == sid].iloc[0]

            col1, col2 = st.columns(2)

            with col1:
                st.subheader(T("Metadata", "Metadatos"))
                elev     = row["elevation"]  if "elevation"  in row.index else "N/A"
                elev_b   = row["elev_band"]  if "elev_band"  in row.index else "N/A"
                mind     = row["mindate"]    if "mindate"    in row.index else "?"
                maxd     = row["maxdate"]    if "maxdate"    in row.index else "?"
                cov_frac = row["cover_frac"] if "cover_frac" in row.index else float("nan")
                scr      = row["score"]      if "score"      in row.index else float("nan")
                st.write(f"**Name:** {row['name']}")
                st.write(f"**ID:** {sid}")
                st.write(f"**Elevation:** {elev} m  ({elev_b} band)")
                st.write(f"**Record:** {mind} → {maxd}")
                st.write(f"**Coverage fraction:** {cov_frac:.2f}")
                st.write(f"**Score:** {scr:.3f}")

            with col2:
                st.subheader(T("Quick quality indicators",
                               "Indicadores rápidos de calidad"))
                missing_prcp = df2["PRCP"].isna().mean() * 100
                missing_tmax = (df2["TMAX"].isna().mean() * 100
                                if "TMAX" in df2 else float("nan"))
                st.markdown(_missing_badge(missing_prcp) + "  *(PRCP)*")
                if not np.isnan(missing_tmax):
                    st.markdown(_missing_badge(missing_tmax) + "  *(TMAX)*")
                st.markdown(_coverage_badge(cov_frac))

                # Gap detection
                df_sorted = df2.sort_values("DATE")
                gaps      = df_sorted["DATE"].diff().dt.days
                big_gaps  = gaps[gaps > 365]
                if not big_gaps.empty:
                    st.warning(T(
                        f"⚠️ {len(big_gaps)} gap(s) > 1 year detected. "
                        "These create discontinuities in SWAT climate inputs.",
                        f"⚠️ {len(big_gaps)} vacío(s) > 1 año detectado(s). "
                        "Crean discontinuidades en las entradas climáticas de SWAT."
                    ))
                else:
                    st.success(T("✅ No gaps > 1 year.",
                                 "✅ Sin vacíos > 1 año."))

                # Interannual CV
                annual_p = df2.groupby("year")["PRCP"].sum()
                cv = (annual_p.std() / annual_p.mean()
                      if annual_p.mean() > 0 else float("nan"))
                st.metric(T("Interannual CV (PRCP)", "CV interanual (PRCP)"),
                          f"{cv:.2f}")

            with st.expander(T("What is the CV and why does it matter?",
                               "¿Qué es el CV y por qué importa?")):
                st.markdown(T(
                    """
                    The **Coefficient of Variation (CV)** measures how variable annual
                    precipitation is relative to its mean.
                    - **CV < 0.3** — Low variability: stable climate, easier to calibrate
                    - **CV 0.3–0.6** — Moderate: typical of semi-arid regions
                    - **CV > 0.6** — High: SWAT needs longer calibration periods to capture
                      the full range of wet and dry conditions; expect lower NSE scores
                    """,
                    """
                    El **Coeficiente de Variación (CV)** mide qué tan variable es la precipitación
                    anual relativa a su media.
                    - **CV < 0.3** — Baja variabilidad: clima estable, más fácil de calibrar
                    - **CV 0.3–0.6** — Moderada: típico de regiones semiáridas
                    - **CV > 0.6** — Alta: SWAT necesita periodos de calibración más largos para
                      capturar el rango completo; espera NSE más bajos
                    """
                ))

            st.subheader(T("Basic statistics", "Estadísticas básicas"))
            st.dataframe(df.describe(), use_container_width=True)

        # ── TAB 2 — TIME SERIES & GAPS ────────────────────────────────────
        with tab2:
            st.subheader(T("Daily Precipitation", "Precipitación Diaria"))

            annual_p    = df2.groupby("year")["PRCP"].sum()
            p25_g, p75_g = annual_p.quantile(0.25), annual_p.quantile(0.75)

            fig_prcp = px.line(df2, x="DATE", y="PRCP",
                               title=T("Daily Precipitation (mm)",
                                       "Precipitación diaria (mm)"))
            st.plotly_chart(fig_prcp, use_container_width=True)

            # Annual bar colored by dry/normal/wet
            annual_df = annual_p.reset_index()
            annual_df["type"] = annual_df["PRCP"].apply(
                lambda v: (T("Dry (< P25)", "Seco (< P25)") if v < p25_g
                           else (T("Wet (> P75)", "Húmedo (> P75)") if v > p75_g
                                 else T("Normal", "Normal")))
            )
            fig_ann = px.bar(
                annual_df, x="year", y="PRCP", color="type",
                color_discrete_map={
                    T("Dry (< P25)", "Seco (< P25)"): "#d62728",
                    T("Normal", "Normal"): "#aec7e8",
                    T("Wet (> P75)", "Húmedo (> P75)"): "#1f77b4",
                },
                title=T("Annual Precipitation — Dry / Normal / Wet",
                        "Precipitación Anual — Seco / Normal / Húmedo")
            )
            st.plotly_chart(fig_ann, use_container_width=True)
            st.caption(T(
                f"P25 = {p25_g:.0f} mm/yr (dry threshold) · P75 = {p75_g:.0f} mm/yr (wet threshold). "
                "A good calibration period includes years from all three categories.",
                f"P25 = {p25_g:.0f} mm/año (umbral seco) · P75 = {p75_g:.0f} mm/año (umbral húmedo). "
                "Un buen periodo de calibración incluye años de las tres categorías."
            ))

            st.subheader(T("Daily Temperature", "Temperatura Diaria"))
            df_temp = df2.melt(id_vars=["DATE"], value_vars=["TMAX", "TMIN"],
                               var_name="variable", value_name="temperature")
            fig_temp = px.line(df_temp, x="DATE", y="temperature",
                               color="variable", title="TMAX / TMIN (°C)")
            st.plotly_chart(fig_temp, use_container_width=True)

            st.subheader(T("Missing Data Heatmap",
                           "Mapa de Calor de Datos Faltantes"))
            missing_df = df2.copy()
            missing_df["missing"] = df2["PRCP"].isna()
            heat = missing_df.pivot_table(index="year", columns="month",
                                          values="missing", aggfunc="mean")
            fig_heat = px.imshow(
                heat, aspect="auto",
                color_continuous_scale="YlOrRd",
                labels=dict(color=T("Fraction missing", "Fracción faltante")),
                title=T("Fraction of missing PRCP by Year / Month",
                        "Fracción de PRCP faltante por Año / Mes")
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            st.caption(T(
                "Dark red = most days missing that month. Systematic patterns "
                "(e.g. always missing in winter) suggest sensor issues, not random gaps.",
                "Rojo oscuro = mayoría de días faltantes ese mes. Patrones sistemáticos "
                "(ej. siempre falta en invierno) sugieren problemas del sensor, no vacíos aleatorios."
            ))

        # ── TAB 3 — CALIBRATION PERIOD ────────────────────────────────────
        with tab3:
            st.subheader(T("Evaluate a calibration window",
                           "Evaluar una ventana de calibración"))

            with st.expander(T("ℹ️ What makes a good calibration period?",
                               "ℹ️ ¿Qué hace un buen periodo de calibración?")):
                st.markdown(T(
                    """
                    A good SWAT calibration period should have:
                    - **Low data missingness** (< 10% ideally)
                    - **Both wet and dry years** — the model must learn to handle the full range
                    - **Sufficient length** — at least 5 years, ideally 10+
                    - **Recent enough** — to reflect current land use and climate patterns

                    The validation period (usually 30–50% of total record) should be a
                    **different** time window the model has never seen.
                    """,
                    """
                    Un buen periodo de calibración de SWAT debe tener:
                    - **Pocos datos faltantes** (< 10% idealmente)
                    - **Años secos Y húmedos** — el modelo debe aprender a manejar todo el rango
                    - **Duración suficiente** — al menos 5 años, idealmente 10+
                    - **Suficientemente reciente** — para reflejar el uso de suelo y clima actuales

                    El periodo de validación (normalmente 30–50% del registro total) debe ser
                    una ventana **diferente** que el modelo nunca haya visto.
                    """
                ))

            min_y = int(df2["year"].min())
            max_y = int(df2["year"].max())
            cal_start_s, cal_end_s = st.slider(
                T("Select calibration years", "Seleccionar años de calibración"),
                min_y, max_y, (min_y, max_y)
            )

            cal_df    = df2[(df2["year"] >= cal_start_s) & (df2["year"] <= cal_end_s)]
            n_years   = cal_end_s - cal_start_s + 1
            miss_pct  = cal_df["PRCP"].isna().mean() * 100
            annual_cal = cal_df.groupby("year")["PRCP"].sum()
            n_dry  = (annual_cal < p25_g).sum()
            n_wet  = (annual_cal > p75_g).sum()
            n_norm = n_years - n_dry - n_wet

            col1, col2, col3, col4 = st.columns(4)
            col1.metric(T("Years", "Años"), n_years)
            col2.metric(T("Missing PRCP", "PRCP faltante"), f"{miss_pct:.1f}%")
            col3.metric(T("Dry years", "Años secos"), n_dry)
            col4.metric(T("Wet years", "Años húmedos"), n_wet)

            # Scorecard
            issues = []
            if n_years < 5:
                issues.append(T("❌ Less than 5 years — too short for reliable calibration.",
                                 "❌ Menos de 5 años — muy corto para calibración confiable."))
            if miss_pct > 15:
                issues.append(T(f"❌ {miss_pct:.1f}% missing — too high. Try a different window.",
                                 f"❌ {miss_pct:.1f}% faltante — muy alto. Prueba otra ventana."))
            if n_dry == 0:
                issues.append(T("⚠️ No dry years — model may not learn drought response.",
                                 "⚠️ Sin años secos — el modelo puede no aprender la respuesta a sequías."))
            if n_wet == 0:
                issues.append(T("⚠️ No wet years — model may underestimate peak flows.",
                                 "⚠️ Sin años húmedos — el modelo puede subestimar caudales pico."))

            if issues:
                for i in issues:
                    st.warning(i)
            else:
                st.success(T(
                    f"✅ Good window: {n_years} yrs · {miss_pct:.1f}% missing · "
                    f"{n_dry} dry + {n_norm} normal + {n_wet} wet years.",
                    f"✅ Buena ventana: {n_years} años · {miss_pct:.1f}% faltante · "
                    f"{n_dry} secos + {n_norm} normales + {n_wet} húmedos."
                ))

            ann_reset = annual_cal.reset_index()
            ann_reset["type"] = ann_reset["PRCP"].apply(
                lambda v: (T("Dry", "Seco") if v < p25_g
                           else (T("Wet", "Húmedo") if v > p75_g
                                 else T("Normal", "Normal")))
            )
            fig_ann2 = px.bar(
                ann_reset, x="year", y="PRCP", color="type",
                color_discrete_map={
                    T("Dry", "Seco"): "#d62728",
                    T("Normal", "Normal"): "#aec7e8",
                    T("Wet", "Húmedo"): "#1f77b4",
                },
                title=T("Annual PRCP in selected calibration window",
                        "PRCP anual en la ventana de calibración seleccionada")
            )
            st.plotly_chart(fig_ann2, use_container_width=True)

        # ── TAB 4 — CLIMATE TRENDS ────────────────────────────────────────
        with tab4:
            with st.expander(T("ℹ️ Why look at trends?",
                               "ℹ️ ¿Por qué ver tendencias?")):
                st.markdown(T(
                    """
                    Climate trends affect SWAT in two important ways:
                    1. **Non-stationarity**: If precipitation or temperature trends strongly,
                       a model calibrated on 2000–2010 may perform poorly on 2020–2030.
                    2. **Validation under a different regime**: If your validation period is
                       drier or warmer, expect lower NSE — this is scientifically normal and
                       important to document, not a sign of a bad model.
                    """,
                    """
                    Las tendencias climáticas afectan a SWAT de dos formas:
                    1. **No-estacionariedad**: Si hay tendencia fuerte en precipitación o temperatura,
                       un modelo calibrado en 2000–2010 puede fallar en 2020–2030.
                    2. **Validación bajo régimen diferente**: Si tu periodo de validación es más
                       seco o cálido, espera NSE más bajos — esto es científicamente normal e
                       importante de documentar, no es un modelo malo.
                    """
                ))

            st.subheader(T("Decadal precipitation", "Precipitación decadal"))
            dec_prcp = df2.groupby("decade")["PRCP"].sum().reset_index()
            fig_dec  = px.bar(dec_prcp, x="decade", y="PRCP",
                              title=T("Total Precipitation by Decade",
                                      "Precipitación Total por Década"))
            st.plotly_chart(fig_dec, use_container_width=True)

            st.subheader(T("Temperature trends", "Tendencias de temperatura"))
            dec_temp     = df2.groupby("decade")[["TMAX", "TMIN"]].mean().reset_index()
            fig_temp_dec = px.line(dec_temp, x="decade", y=["TMAX", "TMIN"],
                                   title=T("Mean TMAX / TMIN by Decade",
                                           "TMAX / TMIN media por Década"))
            st.plotly_chart(fig_temp_dec, use_container_width=True)

            # Linear trend annotation
            annual_prcp_s = df2.groupby("year")["PRCP"].sum().dropna()
            if len(annual_prcp_s) > 3:
                slope, _, r, p_val, _ = linregress(
                    annual_prcp_s.index.astype(float), annual_prcp_s.values
                )
                tpd       = slope * 10
                direction = T("increasing", "creciente") if slope > 0 else T("decreasing", "decreciente")
                sig       = (T("statistically significant (p < 0.05)",
                               "estadísticamente significativa (p < 0.05)")
                             if p_val < 0.05
                             else T("not significant (p ≥ 0.05)",
                                    "no significativa (p ≥ 0.05)"))
                st.info(T(
                    f"📉 Precipitation trend: **{tpd:+.1f} mm/decade** ({direction}), {sig}. R² = {r**2:.2f}.",
                    f"📉 Tendencia de precipitación: **{tpd:+.1f} mm/década** ({direction}), {sig}. R² = {r**2:.2f}."
                ))

        # ══════════════════════════════════════════════════════════════════
        # ADVANCED DIAGNOSTICS — all stations
        # ══════════════════════════════════════════════════════════════════
        st.divider()
        with st.expander(T("📈 Advanced Climate Diagnostics (all stations)",
                           "📈 Diagnósticos Climáticos Avanzados (todas las estaciones)")):

            NOAA_FOLDER = st.text_input(
                T("NOAA data folder", "Carpeta de datos NOAA"), "noaa_raw"
            )

            if not os.path.isdir(NOAA_FOLDER):
                st.warning(T(f"Folder not found: {NOAA_FOLDER}",
                              f"Carpeta no encontrada: {NOAA_FOLDER}"))
                st.stop()

            all_data = load_all_stations(NOAA_FOLDER, folder_sig(NOAA_FOLDER))
            if not all_data:
                st.info(T("No CSV files found in folder.",
                          "No se encontraron archivos CSV en la carpeta."))
                st.stop()

            # Build multi-station metrics table
            metrics      = []
            stations_gdf = st.session_state["stations_thinned"]
            for sid_m, df_m in all_data.items():
                match = stations_gdf[stations_gdf["id"] == sid_m]
                if match.empty:
                    continue
                r = match.iloc[0]
                mp, pt, miss_m, txt, tnt = compute_station_metrics(df_m)
                metrics.append({
                    "id": sid_m, "name": r["name"],
                    "lat": r.geometry.y, "lon": r.geometry.x,
                    "mean_prcp": mp, "prcp_trend": pt,
                    "missing_pct": miss_m,
                    "tmax_trend": txt, "tmin_trend": tnt,
                    "elev_band": r["elev_band"] if "elev_band" in r.index else None,
                })
            metrics_df = pd.DataFrame(metrics)

            adv_tab1, adv_tab2, adv_tab3, adv_tab4 = st.tabs([
                T("🗺️ Spatial Maps",           "🗺️ Mapas Espaciales"),
                T("📊 Multi-Station",           "📊 Multi-Estación"),
                T("📉 Mann–Kendall",            "📉 Mann–Kendall"),
                T("🎯 Calibration Optimizer",   "🎯 Optimizador de Calibración"),
            ])

            # ── Spatial maps ──────────────────────────────────────────────
            with adv_tab1:
                st.caption(T(
                    "Spatial patterns across all stations. Look for outliers — "
                    "unusually dry, wet, or data-sparse stations may need to be "
                    "excluded from SWAT inputs.",
                    "Patrones espaciales en todas las estaciones. Busca atípicos — "
                    "estaciones inusualmente secas, húmedas o con pocos datos pueden "
                    "necesitar excluirse de las entradas de SWAT."
                ))
                for col_name, title, scale in [
                    ("mean_prcp",  T("Mean Annual Precipitation (mm)",
                                     "Precipitación Media Anual (mm)"), "Blues"),
                    ("prcp_trend", T("PRCP Trend (mm/decade)",
                                     "Tendencia PRCP (mm/década)"), "RdBu_r"),
                    ("missing_pct", T("Missing Data (%)",
                                      "Datos Faltantes (%)"), "YlOrRd"),
                    ("tmax_trend", T("TMAX Trend (°C/decade)",
                                     "Tendencia TMAX (°C/década)"), "RdBu_r"),
                    ("tmin_trend", T("TMIN Trend (°C/decade)",
                                     "Tendencia TMIN (°C/década)"), "RdBu_r"),
                ]:
                    st.subheader(title)
                    sz_col = col_name + "_sz"
                    metrics_df[sz_col] = metrics_df[col_name].fillna(0).abs().clip(lower=5)
                    fig = px.scatter_geo(
                        metrics_df, lat="lat", lon="lon",
                        color=col_name, size=sz_col,
                        hover_name="name", projection="natural earth",
                        color_continuous_scale=scale, title=title,
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # ── Multi-station comparison ──────────────────────────────────
            with adv_tab2:
                st.subheader(T("Annual PRCP — All Stations",
                               "PRCP Anual — Todas las Estaciones"))
                st.caption(T(
                    "Stations that diverge strongly from the others in certain years may "
                    "have data quality issues or represent a very different microclimate. "
                    "Investigate before including them in SWAT.",
                    "Estaciones que divergen fuertemente de las demás en ciertos años pueden "
                    "tener problemas de calidad o representar un microclima muy diferente. "
                    "Investiga antes de incluirlas en SWAT."
                ))
                combined = []
                for sid_c, df_c in all_data.items():
                    dfc2 = df_c.copy()
                    dfc2["year"] = dfc2["DATE"].dt.year
                    ann = dfc2.groupby("year")["PRCP"].sum().reset_index()
                    ann["station"] = sid_c
                    combined.append(ann)
                if combined:
                    fig_cmp = px.line(
                        pd.concat(combined), x="year", y="PRCP", color="station",
                        title=T("Annual PRCP by Station", "PRCP Anual por Estación")
                    )
                    st.plotly_chart(fig_cmp, use_container_width=True)

            # ── Mann–Kendall ──────────────────────────────────────────────
            with adv_tab3:
                st.subheader(T("Mann–Kendall Trend Test",
                               "Test de Tendencia Mann–Kendall"))
                with st.expander(T("ℹ️ What is the Mann–Kendall test?",
                                   "ℹ️ ¿Qué es el test de Mann–Kendall?")):
                    st.markdown(T(
                        """
                        The **Mann–Kendall test** detects whether a time series has a
                        monotonic trend, without assuming a normal distribution.

                        - **p < 0.05** — Trend is statistically significant (not random noise)
                        - **Sen's slope** — Median rate of change per year (mm/yr for PRCP)
                        - **Thick-bordered bars** — Significant trends (p < 0.05)

                        If most stations show a significant negative PRCP trend,
                        your watershed is likely experiencing long-term aridification —
                        critical context when interpreting SWAT validation results.
                        """,
                        """
                        El **test de Mann–Kendall** detecta si una serie temporal tiene una
                        tendencia monótona, sin asumir distribución normal.

                        - **p < 0.05** — Tendencia estadísticamente significativa (no ruido aleatorio)
                        - **Pendiente de Sen** — Tasa mediana de cambio por año (mm/año para PRCP)
                        - **Barras con borde grueso** — Tendencias significativas (p < 0.05)

                        Si la mayoría de estaciones muestran tendencia negativa significativa en PRCP,
                        tu cuenca probablemente experimenta aridificación a largo plazo — contexto
                        crítico para interpretar los resultados de validación de SWAT.
                        """
                    ))

                try:
                    import pymannkendall as mk
                except ImportError:
                    st.error("pymannkendall not installed. Run: pip install pymannkendall")
                    st.stop()

                mk_results = []
                for sid_mk, df_mk in all_data.items():
                    dfm2 = df_mk.copy()
                    dfm2["year"] = dfm2["DATE"].dt.year
                    ann_mk = dfm2.groupby("year")["PRCP"].sum()
                    if len(ann_mk) > 5:
                        res = mk.original_test(ann_mk.values)
                        mk_results.append({
                            "station": sid_mk, "trend": res.trend,
                            "p_value": res.p, "slope": res.slope,
                        })

                if mk_results:
                    mk_df = pd.DataFrame(mk_results)
                    mk_df["sign"] = np.where(
                        mk_df["slope"] >= 0,
                        T("Increasing", "Creciente"),
                        T("Decreasing", "Decreciente")
                    )
                    mk_df["significant"]  = mk_df["p_value"] < 0.05
                    mk_df["significance"] = mk_df["significant"].map({
                        True:  T("Significant (p<0.05)", "Significativo (p<0.05)"),
                        False: T("Not significant",      "No significativo"),
                    })
                    st.dataframe(mk_df, use_container_width=True)

                    fig_mk = px.bar(
                        mk_df, x="station", y="slope",
                        color="sign", pattern_shape="significance",
                        color_discrete_map={
                            T("Increasing", "Creciente"): "#1f77b4",
                            T("Decreasing", "Decreciente"): "#d62728",
                        },
                        title=T("Sen's Slope — pattern fill = significant (p<0.05)",
                                "Pendiente de Sen — relleno = significativo (p<0.05)")
                    )
                    mk_df["line_width"] = np.where(mk_df["significant"], 2.5, 0.5)
                    fig_mk.update_traces(
                        marker_line_width=mk_df["line_width"].tolist(),
                        marker_line_color="black"
                    )
                    st.plotly_chart(fig_mk, use_container_width=True)

                    # Auto-interpretation
                    n_sig_neg = ((mk_df["slope"] < 0) & mk_df["significant"]).sum()
                    n_sig_pos = ((mk_df["slope"] > 0) & mk_df["significant"]).sum()
                    if n_sig_neg > len(mk_df) / 2:
                        st.warning(T(
                            f"⚠️ {n_sig_neg}/{len(mk_df)} stations show a significant **decreasing** "
                            "PRCP trend — possible long-term aridification.",
                            f"⚠️ {n_sig_neg}/{len(mk_df)} estaciones muestran tendencia **decreciente** "
                            "significativa — posible aridificación a largo plazo."
                        ))
                    elif n_sig_pos > len(mk_df) / 2:
                        st.info(T(
                            f"ℹ️ {n_sig_pos}/{len(mk_df)} stations show a significant **increasing** PRCP trend.",
                            f"ℹ️ {n_sig_pos}/{len(mk_df)} estaciones muestran tendencia **creciente** significativa."
                        ))
                    else:
                        st.success(T(
                            "✅ No dominant trend direction across stations.",
                            "✅ Sin dirección de tendencia dominante entre estaciones."
                        ))
                else:
                    st.info(T("Not enough data for Mann–Kendall.",
                              "Datos insuficientes para Mann–Kendall."))

            # ── Calibration optimizer ─────────────────────────────────────
            with adv_tab4:
                st.subheader(T("Global Calibration Period Optimizer",
                               "Optimizador Global de Periodo de Calibración"))

                with st.expander(T("ℹ️ How does the optimizer work?",
                                   "ℹ️ ¿Cómo funciona el optimizador?")):
                    st.markdown(T(
                        """
                        The optimizer scores every possible calibration window across all
                        stations simultaneously. The score balances:
                        - **60% data completeness** — windows with lower missingness score higher
                        - **40% climate diversity** — windows containing both dry (< P25) and
                          wet (> P75) years score higher, because they expose SWAT to the full
                          range of hydrological conditions

                        Use the recommended window as a starting point — adjust based on
                        available streamflow records or known land use change dates.
                        """,
                        """
                        El optimizador puntúa todas las ventanas posibles en todas las estaciones.
                        La puntuación balancea:
                        - **60% completitud de datos** — ventanas con menos faltantes tienen mayor puntaje
                        - **40% diversidad climática** — ventanas con años secos (< P25) Y húmedos (> P75)
                          tienen mayor puntaje, porque exponen a SWAT al rango completo de condiciones

                        Usa la ventana recomendada como punto de partida — ajusta según los registros
                        de caudal disponibles o fechas de cambio de uso de suelo conocidas.
                        """
                    ))

                col1, col2 = st.columns(2)
                with col1:
                    min_len = st.number_input(
                        T("Min window (years)", "Ventana mín. (años)"), 3, 50, 5)
                with col2:
                    max_len = st.number_input(
                        T("Max window (years)", "Ventana máx. (años)"), min_len, 60, 20)

                all_years = sorted({
                    yr for df_y in all_data.values()
                    for yr in df_y["year"].dropna().astype(int).unique()
                })

                # Global percentiles for climate diversity
                all_annual_vals = pd.concat([
                    df_y.groupby("year")["PRCP"].sum()
                    for df_y in all_data.values()
                ])
                gp25 = all_annual_vals.quantile(0.25)
                gp75 = all_annual_vals.quantile(0.75)

                best_score, best_window, window_scores = -999.0, None, []

                for start in all_years:
                    for end in all_years:
                        length = end - start + 1
                        if length < min_len or length > max_len or end <= start:
                            continue
                        station_scores = []
                        for df_w in all_data.values():
                            sub = df_w[(df_w["year"] >= start) & (df_w["year"] <= end)]
                            if sub.empty:
                                continue
                            miss_w   = sub["PRCP"].isna().mean()
                            annual_w = sub.groupby("year")["PRCP"].sum()
                            if len(annual_w) < 2:
                                continue
                            diversity = ((annual_w < gp25).sum() +
                                         (annual_w > gp75).sum()) / len(annual_w)
                            score_w = 0.6 * (1 - miss_w) + 0.4 * diversity
                            station_scores.append(score_w)
                        if not station_scores:
                            continue
                        ws = float(np.mean(station_scores))
                        window_scores.append({"start": start, "end": end,
                                              "length": length, "score": ws})
                        if ws > best_score:
                            best_score, best_window = ws, (start, end)

                if best_window is None:
                    st.warning(T("No valid window found with current settings.",
                                 "No se encontró ventana válida con la configuración actual."))
                else:
                    rec_s, rec_e = best_window
                    # Describe recommended window
                    rec_annual = pd.concat([
                        df_w[(df_w["year"] >= rec_s) & (df_w["year"] <= rec_e)
                             ].groupby("year")["PRCP"].sum()
                        for df_w in all_data.values()
                    ])
                    rec_miss = pd.concat([
                        df_w[(df_w["year"] >= rec_s) & (df_w["year"] <= rec_e)]["PRCP"]
                        for df_w in all_data.values()
                    ]).isna().mean() * 100
                    rec_dry = (rec_annual < gp25).sum()
                    rec_wet = (rec_annual > gp75).sum()

                    st.success(T(
                        f"✅ Recommended period: **{rec_s}–{rec_e}**  \n"
                        f"Score: {best_score:.3f} · Missing: {rec_miss:.1f}% · "
                        f"Dry years: {rec_dry} · Wet years: {rec_wet}",
                        f"✅ Periodo recomendado: **{rec_s}–{rec_e}**  \n"
                        f"Score: {best_score:.3f} · Faltante: {rec_miss:.1f}% · "
                        f"Años secos: {rec_dry} · Años húmedos: {rec_wet}"
                    ))

                    scores_df = pd.DataFrame(window_scores)
                    scores_df["center_year"] = (scores_df["start"] + scores_df["end"]) / 2
                    fig_cal = px.scatter(
                        scores_df, x="center_year", y="score", size="length",
                        color="score", color_continuous_scale="Viridis",
                        title=T("Calibration window score (size = duration)",
                                "Puntaje de ventana de calibración (tamaño = duración)"),
                        labels={"center_year": T("Center year", "Año central"),
                                "score": "Score"},
                    )
                    fig_cal.add_vline(
                        x=(rec_s + rec_e) / 2, line_dash="dash", line_color="red",
                        annotation_text=T("Best", "Mejor"),
                    )
                    st.plotly_chart(fig_cal, use_container_width=True)
                    st.caption(T(
                        "Score = 60% data completeness + 40% climate diversity (dry & wet years). "
                        "Larger bubbles = longer windows.",
                        "Score = 60% completitud + 40% diversidad climática (años secos y húmedos). "
                        "Burbujas más grandes = ventanas más largas."
                    ))

        # ══════════════════════════════════════════════════════════════════
        # CLIMATE STRUCTURE VERIFICATION
        # ══════════════════════════════════════════════════════════════════
        st.divider()
        st.header(T("🔬 Climate Structure Verification",
                    "🔬 Verificación de Estructura Climática"))

        with st.expander(T("ℹ️ What is this section for?",
                           "ℹ️ ¿Para qué es esta sección?")):
            st.markdown(T(
                """
                Before running SWAT, verify that your calibration and validation periods
                represent **different but complementary** climate conditions.

                - **Calibration** should cover a range of wet and dry conditions
                - **Validation** should be a genuinely independent period
                - If both periods have identical climate, validation doesn't truly test
                  the model's ability to generalize

                The charts below use **all available stations** to characterize the regional
                climate signal. Green = calibration, red = validation.
                """,
                """
                Antes de correr SWAT, verifica que tus periodos de calibración y validación
                representen condiciones **diferentes pero complementarias**.

                - **Calibración** debe cubrir condiciones húmedas y secas
                - **Validación** debe ser un periodo genuinamente independiente
                - Si ambos periodos tienen clima idéntico, la validación no prueba realmente
                  la capacidad de generalización del modelo

                Las gráficas usan **todas las estaciones disponibles** para caracterizar la
                señal climática regional. Verde = calibración, rojo = validación.
                """
            ))

        NOAA_FOLDER_6B = st.session_state.get("noaa_folder_6b", "noaa_raw")
        df_all = load_all_noaa_daily(NOAA_FOLDER_6B, folder_sig(NOAA_FOLDER_6B))

        if df_all.empty:
            st.warning(T("No data found for climate verification.",
                          "No se encontraron datos para verificación climática."))
        else:
            params_saved = st.session_state.get("params", {})
            cal_s_yr = pd.Timestamp(
                params_saved.get("cal_start", datetime.date(2010, 1, 1))).year
            cal_e_yr = pd.Timestamp(
                params_saved.get("cal_end",   datetime.date(2019, 12, 31))).year
            val_s_yr = cal_e_yr + 1
            val_e_yr = int(df_all["year"].max())

            annual_prcp = (
                df_all.groupby(["year", "DATE"])["PRCP"].mean()
                .groupby("year").sum().reset_index()
            )
            mean_p = annual_prcp["PRCP"].mean()
            std_p  = annual_prcp["PRCP"].std()
            annual_prcp["prcp_anomaly"] = (annual_prcp["PRCP"] - mean_p) / std_p
            prcp_max = annual_prcp["PRCP"].max()

            # Annual precipitation bar
            st.subheader(T("Annual Regional Precipitation",
                           "Precipitación Regional Anual"))
            fig_p = px.bar(
                annual_prcp, x="year", y="PRCP",
                labels={"PRCP": T("Annual precipitation (mm)",
                                   "Precipitación anual (mm)")}
            )
            fig_p.add_vrect(x0=cal_s_yr, x1=cal_e_yr,
                            fillcolor="green", opacity=0.15, layer="below")
            fig_p.add_vrect(x0=val_s_yr, x1=val_e_yr,
                            fillcolor="red", opacity=0.15, layer="below")
            fig_p.add_annotation(
                x=(cal_s_yr + cal_e_yr) / 2, y=prcp_max * 0.95,
                text=T(f"Calibration ({cal_s_yr}–{cal_e_yr})",
                       f"Calibración ({cal_s_yr}–{cal_e_yr})"),
                showarrow=False
            )
            fig_p.add_annotation(
                x=(val_s_yr + val_e_yr) / 2, y=prcp_max * 0.95,
                text=T(f"Validation ({val_s_yr}–{val_e_yr})",
                       f"Validación ({val_s_yr}–{val_e_yr})"),
                showarrow=False
            )
            st.plotly_chart(fig_p, use_container_width=True)

            # Standardized anomalies
            st.subheader(T("Standardized Precipitation Anomalies",
                           "Anomalías Estandarizadas de Precipitación"))
            fig_a = px.bar(
                annual_prcp, x="year", y="prcp_anomaly",
                color="prcp_anomaly", color_continuous_scale="RdBu",
                labels={"prcp_anomaly": "z-score"}
            )
            fig_a.add_hline(y=0, line_dash="dash")
            fig_a.add_vrect(x0=cal_s_yr, x1=cal_e_yr,
                            fillcolor="green", opacity=0.12, layer="below")
            fig_a.add_vrect(x0=val_s_yr, x1=val_e_yr,
                            fillcolor="red", opacity=0.12, layer="below")
            st.plotly_chart(fig_a, use_container_width=True)
            st.caption(T(
                "Negative z-scores = dry years, positive = wet years. "
                "Ideally calibration spans both positive and negative anomalies.",
                "Z-scores negativos = años secos, positivos = húmedos. "
                "Idealmente la calibración abarca anomalías positivas y negativas."
            ))

            # Mean temperature
            df_all["TMEAN"] = (df_all["TMAX"] + df_all["TMIN"]) / 2
            annual_temp = df_all.groupby("year")["TMEAN"].mean().reset_index()
            st.subheader(T("Mean Annual Temperature", "Temperatura Media Anual"))
            fig_t = px.line(
                annual_temp, x="year", y="TMEAN",
                labels={"TMEAN": T("Mean temperature (°C)", "Temperatura media (°C)")}
            )
            fig_t.add_vrect(x0=cal_s_yr, x1=cal_e_yr,
                            fillcolor="green", opacity=0.15, layer="below")
            fig_t.add_vrect(x0=val_s_yr, x1=val_e_yr,
                            fillcolor="red", opacity=0.15, layer="below")
            st.plotly_chart(fig_t, use_container_width=True)

            # ── Automatic interpretation ───────────────────────────────────
            st.subheader(T("🧠 Automatic interpretation",
                           "🧠 Interpretación automática"))

            cal_anom = annual_prcp[
                (annual_prcp["year"] >= cal_s_yr) &
                (annual_prcp["year"] <= cal_e_yr)
            ]["prcp_anomaly"].mean()
            val_anom = annual_prcp[
                (annual_prcp["year"] >= val_s_yr) &
                (annual_prcp["year"] <= val_e_yr)
            ]["prcp_anomaly"].mean()
            cal_temp = annual_temp[
                (annual_temp["year"] >= cal_s_yr) &
                (annual_temp["year"] <= cal_e_yr)
            ]["TMEAN"].mean()
            val_temp = annual_temp[
                (annual_temp["year"] >= val_s_yr) &
                (annual_temp["year"] <= val_e_yr)
            ]["TMEAN"].mean()

            if val_anom < cal_anom - 0.3:
                st.warning(T(
                    f"⚠️ Validation period is **drier** than calibration "
                    f"(anomaly {val_anom:.2f} vs {cal_anom:.2f}). "
                    "Expect lower NSE during validation — this is scientifically normal, "
                    "not a model failure. Document it explicitly in your paper.",
                    f"⚠️ El periodo de validación es **más seco** que la calibración "
                    f"(anomalía {val_anom:.2f} vs {cal_anom:.2f}). "
                    "Espera NSE más bajo en validación — esto es científicamente normal, "
                    "no un fallo del modelo. Documéntalo explícitamente en tu artículo."
                ))
            elif val_anom > cal_anom + 0.3:
                st.info(T(
                    f"ℹ️ Validation period is **wetter** than calibration "
                    f"(anomaly {val_anom:.2f} vs {cal_anom:.2f}).",
                    f"ℹ️ El periodo de validación es **más húmedo** que la calibración "
                    f"(anomalía {val_anom:.2f} vs {cal_anom:.2f})."
                ))
            else:
                st.success(T(
                    "✅ Calibration and validation have similar precipitation anomalies — good structure.",
                    "✅ Calibración y validación tienen anomalías similares — buena estructura."
                ))

            if not (np.isnan(val_temp) or np.isnan(cal_temp)):
                dt = val_temp - cal_temp
                if dt > 0.5:
                    st.warning(T(
                        f"🌡️ Validation period is **{dt:.1f}°C warmer** than calibration. "
                        "Document this as validation under non-stationary conditions.",
                        f"🌡️ El periodo de validación es **{dt:.1f}°C más cálido** que la calibración. "
                        "Documenta esto como validación bajo condiciones no estacionarias."
                    ))
                elif dt < -0.5:
                    st.info(T(
                        f"🌡️ Validation period is **{abs(dt):.1f}°C cooler** than calibration.",
                        f"🌡️ El periodo de validación es **{abs(dt):.1f}°C más frío** que la calibración."
                    ))
                else:
                    st.success(T(
                        f"✅ Temperature difference between periods: {dt:+.2f}°C — negligible.",
                        f"✅ Diferencia de temperatura entre periodos: {dt:+.2f}°C — despreciable."
                    ))

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5 — Export SWAT+ ready files
    # ══════════════════════════════════════════════════════════════════════════
    elif step == "5️⃣ Export SWAT+":

        st.header(T("5. Export SWAT+ Climate Files",
                    "5. Exportar Archivos Climáticos para SWAT+"))

        if "stations_thinned" not in st.session_state:
            st.warning(T("Complete Steps 1–3 first to select stations.",
                         "Completa los Pasos 1–3 primero para seleccionar estaciones."))
            st.stop()

        with st.expander(T("ℹ️ What files will be generated?",
                           "ℹ️ ¿Qué archivos se generarán?"), expanded=True):
            st.markdown(T(
                """
                This step generates all climate input files needed to run **SWAT+**
                with your selected NOAA stations:

                | File | Description |
                |---|---|
                | `<station>.pcp` | Daily precipitation per station (mm) |
                | `<station>.tmp` | Daily TMAX / TMIN per station (°C) |
                | `pcp.cli` | Index of all precipitation files |
                | `tmp.cli` | Index of all temperature files |
                | `weather-sta.cli` | Station list with lat/lon/elev for SWAT+ |
                | `stations.shp` | Point shapefile of selected stations |

                > 💡 **Missing values** are written as `-99.0`, which tells SWAT+
                > to use the weather generator for that day — no manual gap-filling needed.

                > 📂 **How to use in SWAT+:** Copy all `.pcp`, `.tmp` and `.cli` files
                > into your SWAT+ project `Scenarios/Default/TxtInOut/` folder.
                > The shapefile can be loaded in QSWAT+ to verify station locations.
                """,
                """
                Este paso genera todos los archivos de entrada climática para correr **SWAT+**
                con tus estaciones NOAA seleccionadas:

                | Archivo | Descripción |
                |---|---|
                | `<estacion>.pcp` | Precipitación diaria por estación (mm) |
                | `<estacion>.tmp` | TMAX / TMIN diarios por estación (°C) |
                | `pcp.cli` | Índice de todos los archivos de precipitación |
                | `tmp.cli` | Índice de todos los archivos de temperatura |
                | `weather-sta.cli` | Lista de estaciones con lat/lon/elev para SWAT+ |
                | `stations.shp` | Shapefile de puntos de las estaciones seleccionadas |

                > 💡 **Datos faltantes** se escriben como `-99.0`, lo que indica a SWAT+
                > que use el generador climático para ese día — no se necesita relleno manual.

                > 📂 **Cómo usar en SWAT+:** Copia todos los archivos `.pcp`, `.tmp` y `.cli`
                > a la carpeta `Scenarios/Default/TxtInOut/` de tu proyecto SWAT+.
                > El shapefile se puede cargar en QSWAT+ para verificar las ubicaciones.
                """
            ))

        stations_thinned = st.session_state["stations_thinned"]
        params           = st.session_state.get("params", {})
        n_stations       = len(stations_thinned)

        # ── Configuration ──────────────────────────────────────────────────
        st.subheader(T("⚙️ Export settings", "⚙️ Configuración de exportación"))

        col1, col2, col3 = st.columns(3)
        with col1:
            noaa_folder_exp = st.text_input(
                T("NOAA data folder", "Carpeta de datos NOAA"),
                value="noaa_raw", key="exp_noaa_folder"
            )
        with col2:
            exp_start = st.date_input(
                T("Export start date", "Fecha de inicio de exportación"),
                value=params.get("cal_start", datetime.date(2000, 1, 1)),
                min_value=EARLIEST_DATE, max_value=datetime.date(2100, 12, 31),
                key="exp_start"
            )
        with col3:
            exp_end = st.date_input(
                T("Export end date", "Fecha de fin de exportación"),
                value=params.get("cal_end", datetime.date(2024, 12, 31)),
                min_value=EARLIEST_DATE, max_value=datetime.date(2100, 12, 31),
                key="exp_end"
            )

        output_folder = st.text_input(
            T("Output folder for SWAT+ files", "Carpeta de salida para archivos SWAT+"),
            value="swatplus_climate", key="exp_output_folder"
        )

        # ── Pre-flight check ───────────────────────────────────────────────
        st.subheader(T("🔍 Pre-flight check", "🔍 Verificación previa"))

        existing_csvs = set()
        if os.path.isdir(noaa_folder_exp):
            existing_csvs = {f.replace(".csv", "").replace("_", ":")
                             for f in os.listdir(noaa_folder_exp)
                             if f.endswith(".csv")}

        station_ids   = stations_thinned["id"].tolist() if "id" in stations_thinned.columns else []
        ready         = [s for s in station_ids if s in existing_csvs]
        not_ready     = [s for s in station_ids if s not in existing_csvs]

        col1, col2 = st.columns(2)
        col1.metric(T("✅ Stations ready to export", "✅ Estaciones listas para exportar"),
                    len(ready))
        col2.metric(T("❌ Missing data files", "❌ Archivos de datos faltantes"),
                    len(not_ready))

        if not_ready:
            st.warning(T(
                f"⚠️ {len(not_ready)} station(s) have no local CSV and will be skipped. "
                "Go to **3️⃣ Results** → *Daily Data Status* to download them first.",
                f"⚠️ {len(not_ready)} estación(es) no tienen CSV local y serán omitidas. "
                "Ve a **3️⃣ Results** → *Estado de Datos Diarios* para descargarlas primero."
            ))
            with st.expander(T("Show missing stations", "Mostrar estaciones faltantes")):
                st.write(not_ready)

        if not ready:
            st.error(T(
                "No station data available. Download data in Step 3 first.",
                "Sin datos de estaciones disponibles. Descarga datos en el Paso 3 primero."
            ))
            st.stop()

        # ── Export button ──────────────────────────────────────────────────
        exp_years = (pd.Timestamp(exp_end) - pd.Timestamp(exp_start)).days / 365.25
        st.info(T(
            f"Ready to export **{len(ready)} stations** · "
            f"{exp_start} → {exp_end} ({exp_years:.1f} years) · "
            f"Output: `{output_folder}/`",
            f"Listo para exportar **{len(ready)} estaciones** · "
            f"{exp_start} → {exp_end} ({exp_years:.1f} años) · "
            f"Salida: `{output_folder}/`"
        ))

        if st.button(T("🚀 Generate SWAT+ files", "🚀 Generar archivos SWAT+"),
                     type="primary"):

            with st.spinner(T("Generating SWAT+ climate files…",
                               "Generando archivos climáticos SWAT+…")):
                try:
                    result = export_swatplus(
                        stations_gdf=stations_thinned,
                        noaa_folder=noaa_folder_exp,
                        output_folder=output_folder,
                        start_date=exp_start,
                        end_date=exp_end,
                    )
                except Exception as e:
                    st.error(T(f"Export failed: {e}", f"Exportación fallida: {e}"))
                    st.stop()

            # ── Results ────────────────────────────────────────────────────
            st.success(T(
                f"✅ Export complete!  \n"
                f"- {result['n_pcp']} `.pcp` files  \n"
                f"- {result['n_tmp']} `.tmp` files  \n"
                f"- `pcp.cli`, `tmp.cli`, `weather-sta.cli`  \n"
                f"- `stations.shp` (+ .shx .dbf .prj)  \n"
                f"- All packed in `{os.path.basename(result['zip_path'])}`",
                f"✅ ¡Exportación completa!  \n"
                f"- {result['n_pcp']} archivos `.pcp`  \n"
                f"- {result['n_tmp']} archivos `.tmp`  \n"
                f"- `pcp.cli`, `tmp.cli`, `weather-sta.cli`  \n"
                f"- `stations.shp` (+ .shx .dbf .prj)  \n"
                f"- Todo empaquetado en `{os.path.basename(result['zip_path'])}`"
            ))

            if result["skipped"]:
                st.warning(T(
                    f"⚠️ {len(result['skipped'])} station(s) skipped (no local CSV): "
                    f"{result['skipped']}",
                    f"⚠️ {len(result['skipped'])} estación(es) omitidas (sin CSV local): "
                    f"{result['skipped']}"
                ))

            # ── File preview ───────────────────────────────────────────────
            with st.expander(T("👁️ Preview generated files",
                               "👁️ Vista previa de archivos generados")):

                out_files = sorted(os.listdir(result["output_dir"]))

                # Show pcp.cli
                cli_path = os.path.join(result["output_dir"], "pcp.cli")
                if os.path.exists(cli_path):
                    st.markdown("**`pcp.cli`**")
                    with open(cli_path) as f:
                        st.code(f.read(), language="text")

                # Show weather-sta.cli
                wsta_path = os.path.join(result["output_dir"], "weather-sta.cli")
                if os.path.exists(wsta_path):
                    st.markdown("**`weather-sta.cli`**")
                    with open(wsta_path) as f:
                        st.code(f.read(), language="text")

                # Show first 10 lines of first .pcp file
                pcp_files_list = [f for f in out_files if f.endswith(".pcp")]
                if pcp_files_list:
                    sample_path = os.path.join(result["output_dir"], pcp_files_list[0])
                    st.markdown(f"**`{pcp_files_list[0]}` (first 10 data lines)**")
                    with open(sample_path) as f:
                        lines = f.readlines()
                    st.code("".join(lines[:13]), language="text")

            # ── Download zip ───────────────────────────────────────────────
            st.subheader(T("⬇️ Download all files", "⬇️ Descargar todos los archivos"))

            with open(result["zip_path"], "rb") as zf:
                zip_bytes = zf.read()

            st.download_button(
                label=T(
                    f"⬇️ Download SWAT+ climate package ({len(zip_bytes)//1024:,} KB)",
                    f"⬇️ Descargar paquete climático SWAT+ ({len(zip_bytes)//1024:,} KB)"
                ),
                data=zip_bytes,
                file_name=os.path.basename(result["zip_path"]),
                mime="application/zip",
            )

            # ── Next steps guidance ────────────────────────────────────────
            st.subheader(T("📋 Next steps in SWAT+",
                           "📋 Próximos pasos en SWAT+"))
            st.markdown(T(
                """
                1. **Unzip** the downloaded package
                2. Copy **all `.pcp` and `.tmp` files** → `Scenarios/Default/TxtInOut/`
                3. Copy **`pcp.cli` and `tmp.cli`** → same `TxtInOut/` folder
                4. Copy **`weather-sta.cli`** → same `TxtInOut/` folder
                5. In **SWAT+ Editor**: go to *Climate* → *Weather Stations* →
                   import `weather-sta.cli`
                6. Load **`stations.shp`** in **QGIS / QSWAT+** to verify spatial coverage
                   before running the model
                7. Set your simulation period to match the export dates:
                   **{exp_start} → {exp_end}**

                > ⚠️ Make sure the simulation period in SWAT+ does not extend **beyond**
                > the date range of your downloaded data, or SWAT+ will rely entirely
                > on the weather generator for those periods.
                """.format(exp_start=exp_start, exp_end=exp_end),
                """
                1. **Descomprime** el paquete descargado
                2. Copia **todos los archivos `.pcp` y `.tmp`** → `Scenarios/Default/TxtInOut/`
                3. Copia **`pcp.cli` y `tmp.cli`** → misma carpeta `TxtInOut/`
                4. Copia **`weather-sta.cli`** → misma carpeta `TxtInOut/`
                5. En **SWAT+ Editor**: ve a *Climate* → *Weather Stations* →
                   importa `weather-sta.cli`
                6. Carga **`stations.shp`** en **QGIS / QSWAT+** para verificar cobertura
                   espacial antes de correr el modelo
                7. Ajusta el periodo de simulación para que coincida con las fechas de exportación:
                   **{exp_start} → {exp_end}**

                > ⚠️ Asegúrate de que el periodo de simulación en SWAT+ no se extienda
                > **más allá** del rango de fechas de tus datos descargados, o SWAT+
                > usará completamente el generador climático para esos periodos.
                """.format(exp_start=exp_start, exp_end=exp_end)
            ))


if __name__ == "__main__":
    run_app()
