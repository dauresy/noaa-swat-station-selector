"""
Keep a user's work (basin, parameters, station selection) on the server so it
survives a Streamlit session reset (reconnect, app restart, page reload).

Each browser gets a random code in the URL (?s=...). Everything is stored under
that code in the system temp folder as Parquet/JSON (no pickle). Anyone with the
link can restore that work, so the code is long and random; the NOAA API token
is never written to disk.
"""
import datetime
import json
import re
import secrets
import shutil
import tempfile
import time
from pathlib import Path

import geopandas as gpd

ROOT = Path(tempfile.gettempdir()) / "noaa_selector_sessions"
_TOKEN_RE = re.compile(r"^[0-9a-f]{16}$")
_GEO_PARTS = {"basin", "stations", "buffer"}      # stored as <name>.parquet
_JSON_PARTS = {"params", "meta"}                  # stored as <name>.json
MAX_AGE_DAYS = 7


def new_token() -> str:
    return secrets.token_hex(8)


def valid_token(token) -> bool:
    return isinstance(token, str) and bool(_TOKEN_RE.match(token))


def _dir(token: str) -> Path:
    if not valid_token(token):
        raise ValueError("invalid session code")
    return ROOT / token


def _json_default(o):
    if isinstance(o, (datetime.date, datetime.datetime)):
        return {"__date__": o.isoformat()}
    raise TypeError(type(o))


def _json_hook(d):
    return datetime.date.fromisoformat(d["__date__"]) if "__date__" in d else d


def save_part(token: str, name: str, obj) -> None:
    d = _dir(token)
    d.mkdir(parents=True, exist_ok=True)
    if name in _GEO_PARTS:
        tmp = d / f"{name}.parquet.tmp"
        try:
            obj.to_parquet(tmp)
        except Exception:
            # odd attribute dtypes (mixed object columns): keep the geometry only
            gpd.GeoDataFrame(geometry=obj.geometry, crs=obj.crs).to_parquet(tmp)
        tmp.replace(d / f"{name}.parquet")
    elif name in _JSON_PARTS:
        tmp = d / f"{name}.json.tmp"
        tmp.write_text(json.dumps(obj, default=_json_default), encoding="utf-8")
        tmp.replace(d / f"{name}.json")
    else:
        raise ValueError(name)


def delete_part(token: str, name: str) -> None:
    d = _dir(token)
    for ext in ("parquet", "json"):
        (d / f"{name}.{ext}").unlink(missing_ok=True)


def delete_session(token: str) -> None:
    shutil.rmtree(_dir(token), ignore_errors=True)


def load_all(token: str) -> dict:
    """Return whatever parts exist for this code ({} if none)."""
    if not valid_token(token):
        return {}
    d = _dir(token)
    out = {}
    if not d.is_dir():
        return out
    for name in _GEO_PARTS:
        f = d / f"{name}.parquet"
        if f.exists():
            try:
                out[name] = gpd.read_parquet(f)
            except Exception:
                pass
    for name in _JSON_PARTS:
        f = d / f"{name}.json"
        if f.exists():
            try:
                out[name] = json.loads(f.read_text(encoding="utf-8"), object_hook=_json_hook)
            except Exception:
                pass
    # touch so an active session is not purged
    try:
        d.touch()
    except OSError:
        pass
    return out


def purge_old(days: int = MAX_AGE_DAYS) -> None:
    if not ROOT.is_dir():
        return
    limit = time.time() - days * 86400
    for d in ROOT.iterdir():
        try:
            if d.is_dir() and d.stat().st_mtime < limit:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass
