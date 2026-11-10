# 🌧️ NOAA GHCND Station Selector for SWAT+

> **A Streamlit app for hydrologists and watershed modelers.**  
> Upload any watershed shapefile → automatically find, filter, and download the best NOAA climate stations → export SWAT+-ready `.pcp` and `.tmp` files in one click.

[![Live App](https://img.shields.io/badge/🚀%20Live%20App-noaa--station--selector.streamlit.app-FF4B4B?logo=streamlit&logoColor=white)](https://noaa-station-selector.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**👉 [Open the live app](https://noaa-station-selector.streamlit.app/) — no installation needed.**

---

## Why this tool?

Choosing the right climate stations is one of the most critical and time-consuming steps in a SWAT+ modeling workflow. A station with 40% missing data or no overlap with your calibration window introduces errors that **no parameter tuning can fix**.

This app automates the entire process:

| Step | What it does |
|---|---|
| **1. Upload** | Load your watershed shapefile (`.shp + .shx + .dbf + .prj`) |
| **2. Parameters** | Set buffer, calibration window, quality thresholds |
| **3. Results** | Automatically downloads NOAA stations, scores them, thins spatially |
| **4. Analysis** | Per-station data quality, missing data heatmaps, climate trends, Mann–Kendall, calibration optimizer |
| **5. Export** | One-click export of all SWAT+ climate files (`.pcp`, `.tmp`, `.cli`, `weather-sta.cli`, `.shp`) |

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/JosephAuresy/noaa-swat-station-selector.git
cd noaa-swat-station-selector

# 2. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

---

## NOAA API Token

The station **search** uses the **NOAA Climate Data Online (CDO) API**, which requires a free token.

**Get your free token here (30 seconds):**  
👉 https://www.ncdc.noaa.gov/cdo-web/token

Paste it in the app when prompted. The token is never saved to disk.

The **daily data download** does not need the token: each station's full GHCN-Daily record is fetched
in one request from NOAA's public file server, so any start date from **1900** onwards works
(the CDO API would need one request per year and is capped at 10,000 requests a day).
The token is only used as a fallback if that server is unreachable.

## Works anywhere in the U.S.

Buffers, station spacing and basin area are computed in a projection chosen from the basin's own
location, so basins outside Texas/New Mexico are measured correctly.

## Keeping the live app awake

Streamlit Community Cloud puts apps to sleep after 12 hours without traffic. The workflow in
`.github/workflows/keepalive.yml` opens the app in a headless browser every 6 hours (and clicks
"Yes, get this app back up!" if it finds the app asleep). It targets
`https://noaa-station-selector.streamlit.app/`; set a repository variable `STREAMLIT_URL` to change it.

---

## Station scoring algorithm

Each candidate station receives a composite **quality score** (0–1):

```
score = 0.4 × span_norm          # longer record = better
      + 0.4 × cover_frac         # covers more of your calibration window = better
      + 0.2 × recency_norm       # still active recently = better
```

When two stations are within the minimum spacing distance, the one with the **higher score is kept** — thinning is never random.

---

## Output files

After running the export step you get a `.zip` with:

```
swatplus_climate/
├── GHCND_USC00123456.pcp     # Daily precipitation per station
├── GHCND_USC00123456.tmp     # Daily TMAX / TMIN per station
├── pcp.cli                   # Index of all .pcp files
├── tmp.cli                   # Index of all .tmp files
├── weather-sta.cli            # Station list (lat / lon / elev) for SWAT+
└── stations.shp              # Point shapefile of selected stations
```

Missing values are written as `-99.0`, which tells SWAT+ to use the weather generator — no manual gap-filling needed.

### How to use in SWAT+

1. Unzip the downloaded package
2. Copy all `.pcp` and `.tmp` files → `Scenarios/Default/TxtInOut/`
3. Copy `pcp.cli`, `tmp.cli`, `weather-sta.cli` → same folder
4. In **SWAT+ Editor**: *Climate → Weather Stations → import `weather-sta.cli`*
5. Load `stations.shp` in **QGIS / QSWAT+** to verify spatial coverage

---

## Project structure

```
noaa-swat-station-selector/
├── app.py                        # Main Streamlit entry point
├── scripts/
│   ├── select_stations.py        # Download + filter + score pipeline
│   ├── filter_and_score.py       # Quality filters & spatial thinning
│   ├── download_noaa.py          # Station metadata from CDO API
│   ├── download_noaa_daily.py    # Daily data download wrapper
│   ├── download_noaa_yearly.py   # Year-by-year data fetcher
│   ├── export_swatplus.py        # SWAT+ file writers
│   ├── analyze_subbasins.py      # Subbasin–station analysis
│   └── ...
├── requirements.txt
└── .streamlit/config.toml
```

---

## Requirements

- Python 3.10+
- Internet connection (for NOAA CDO API calls)
- Free NOAA API token (see above)

Key dependencies: `streamlit`, `geopandas`, `folium`, `plotly`, `scipy`, `requests`

---

## Related project

This tool is part of the **Texas Pecos Watershed Climate Dashboard** developed at the  
[Texas Water Consortium (TXPWC)](https://www.depts.ttu.edu/research/tx-water-consortium/) — Texas Tech University.

The dashboard version is pre-loaded for the Pecos River basin and does not require API calls.

---

## Citation

If you use this tool in a publication, please cite:

```
Serrano Suarez, D. (2025). NOAA GHCND Station Selector for SWAT+.
GitHub: https://github.com/JosephAuresy/noaa-swat-station-selector
```

---

## License

MIT — free to use, modify, and distribute with attribution.
