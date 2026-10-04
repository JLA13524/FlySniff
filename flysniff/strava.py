"""Get activities from Strava and boil each one down to a few features.

Three sources:
  * a bulk-export `activities.csv` (Strava > Settings > My Account > Download your data)
  * the Strava API, with STRAVA_ACCESS_TOKEN set (scope: activity:read_all)
  * `demo_activities()` – a fake fortnight of triathlon training
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

INTERVAL_WORDS = re.compile(r"interval|reps|vo2|threshold|x\s*\d|\d+\s*x|track|fartlek|tempo|sweet ?spot", re.I)

FEATURES = ["speed", "easy", "cadence", "hr_drift", "climb", "intervalness"]


def _col(df: pd.DataFrame, *names: str) -> pd.Series:
    """Strava's CSV has duplicate/renamed headers; take the first that has data."""
    for n in names:
        for c in df.columns:
            if c.split(".")[0].strip().lower() == n.lower():
                s = pd.to_numeric(df[c], errors="coerce")
                if s.notna().any():
                    return s
    return pd.Series(np.nan, index=df.index)


def load_csv(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    out = pd.DataFrame({
        "id": raw.get("Activity ID", pd.Series(range(len(raw)))).astype(str),
        "name": raw.get("Activity Name", "").fillna(""),
        "type": raw.get("Activity Type", "").fillna(""),
        "date": pd.to_datetime(raw.get("Activity Date"), errors="coerce", format="mixed"),
        "moving_s": _col(raw, "Moving Time", "Elapsed Time"),
        "speed_ms": _col(raw, "Average Speed"),
        "distance_m": _col(raw, "Distance"),
        "elev_m": _col(raw, "Elevation Gain"),
        "avg_hr": _col(raw, "Average Heart Rate"),
        "max_hr": _col(raw, "Max Heart Rate"),
        "cadence": _col(raw, "Average Cadence"),
        "kudos": np.nan,
        "speed_cv": np.nan,
    })
    # CSV distance is km for the first column; derive speed if missing.
    km = out.distance_m < 1000
    out.loc[km, "distance_m"] *= 1000
    miss = out.speed_ms.isna() & (out.moving_s > 0)
    out.loc[miss, "speed_ms"] = out.distance_m / out.moving_s
    out["drift_pct"] = np.nan
    return out


def load_api(token: str, n: int = 60, streams: bool = False) -> pd.DataFrame:
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    acts, page = [], 1
    while len(acts) < n:
        r = s.get("https://www.strava.com/api/v3/athlete/activities",
                  params={"per_page": min(100, n), "page": page}, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        acts += batch
        page += 1
    acts = acts[:n]
    rows = []
    for a in acts:
        drift = np.nan
        if streams and a.get("has_heartrate"):
            drift = _decoupling(s, a["id"])
        rows.append({
            "id": str(a["id"]), "name": a.get("name", ""), "type": a.get("sport_type", a.get("type", "")),
            "date": pd.to_datetime(a.get("start_date_local")), "moving_s": a.get("moving_time"),
            "speed_ms": a.get("average_speed"), "distance_m": a.get("distance"),
            "elev_m": a.get("total_elevation_gain"), "avg_hr": a.get("average_heartrate"),
            "max_hr": a.get("max_heartrate"), "cadence": a.get("average_cadence"),
            "kudos": a.get("kudos_count"), "drift_pct": drift, "speed_cv": np.nan,
        })
    return pd.DataFrame(rows)


def _decoupling(s: requests.Session, act_id) -> float:
    """Pa:HR decoupling – speed/HR in the 2nd half vs the 1st, as a %."""
    r = s.get(f"https://www.strava.com/api/v3/activities/{act_id}/streams",
              params={"keys": "heartrate,velocity_smooth", "key_by_type": "true"}, timeout=30)
    if r.status_code != 200:
        return np.nan
    j = r.json()
    hr = np.asarray(j.get("heartrate", {}).get("data", []), float)
    v = np.asarray(j.get("velocity_smooth", {}).get("data", []), float)
    m = min(len(hr), len(v))
    if m < 60:
        return np.nan
    hr, v = hr[:m], v[:m]
    h = m // 2
    e1 = v[:h].mean() / max(hr[:h].mean(), 1)
    e2 = v[h:].mean() / max(hr[h:].mean(), 1)
    return float((e1 - e2) / e1 * 100)


def demo_activities(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2026, 9, 14, 7, 0)
    plan = [  # (day offset, name, type, km, min/km or kph, elev, hr, maxhr, cad)
        (0, "Easy recovery jog", "Run", 7, 5.9, 35, 132, 148, 168),
        (1, "Tuesday track intervals 6x800", "Run", 10, 4.6, 20, 158, 186, 182),
        (2, "Sweet spot turbo", "Ride", 40, 32, 0, 148, 168, 92),
        (3, "Lunch swim", "Swim", 2.5, 2.0, 0, 128, 150, 0),
        (4, "Tempo run along the canal", "Run", 12, 4.8, 40, 154, 172, 178),
        (5, "Long hilly ride – Chilterns", "Ride", 105, 27, 1350, 141, 176, 84),
        (6, "Long Sunday run", "Run", 21, 5.3, 160, 146, 165, 172),
        (7, "Easy shakeout", "Run", 6, 6.0, 20, 128, 140, 166),
        (8, "Tuesday VO2 intervals 5x1k", "Run", 11, 4.5, 25, 161, 189, 184),
        (9, "Z2 endurance ride", "Ride", 60, 29, 380, 132, 150, 88),
        (10, "Open water swim", "Swim", 3, 2.1, 0, 131, 152, 0),
        (11, "Parkrun PB attempt", "Run", 5, 3.9, 30, 171, 190, 186),
        (12, "Brick: bike + run off", "Ride", 50, 31, 420, 150, 174, 90),
        (13, "Hill repeats", "Run", 9, 5.0, 310, 159, 185, 176),
    ]
    rows = []
    for i, (d, name, typ, km, pace, elev, hr, mx, cad) in enumerate(plan):
        if typ == "Ride":
            speed = pace / 3.6
        elif typ == "Swim":
            speed = 100 / (pace * 60)          # pace is min/100m
        else:
            speed = 1000 / (pace * 60)          # pace is min/km
        moving = km * 1000 / speed
        drift = {"Long Sunday run": 7.5, "Long hilly ride – Chilterns": 9.0, "Hill repeats": 6.5,
                 "Brick: bike + run off": 6.0, "Tuesday VO2 intervals 5x1k": 5.5,
                 "Tuesday track intervals 6x800": 5.0}.get(name, rng.normal(1.5, 0.6))
        rows.append({
            "id": f"demo{i}", "name": name, "type": typ,
            "date": pd.Timestamp(start + timedelta(days=d, minutes=int(rng.integers(0, 600)))),
            "moving_s": moving, "speed_ms": speed, "distance_m": km * 1000, "elev_m": elev,
            "avg_hr": hr, "max_hr": mx, "cadence": cad or np.nan,
            "kudos": int(rng.integers(2, 25)), "drift_pct": drift,
            "speed_cv": 0.22 if ("interval" in name.lower() or "repeats" in name.lower()) else 0.06,
        })
    return pd.DataFrame(rows)


def _z(x: pd.Series) -> pd.Series:
    sd = x.std()
    return (x - x.mean()) / sd if sd and sd > 0 else x * 0


def raw_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """The per-activity numbers the features are built from. Written into
    results.json so the web page can re-score new uploads against them."""
    r = pd.DataFrame(index=df.index)
    r["type"] = df.type.fillna("").astype(str)
    for c in ["speed_ms", "avg_hr", "max_hr", "cadence", "drift_pct"]:
        r[c] = pd.to_numeric(df[c], errors="coerce")
    r["climb"] = df.elev_m / (df.distance_m / 1000).clip(lower=1)   # m per km
    r["named"] = df.name.fillna("").str.contains(INTERVAL_WORDS).astype(float)
    r["speed_cv"] = pd.to_numeric(df.get("speed_cv", np.nan), errors="coerce")
    return r


def features(df: pd.DataFrame) -> pd.DataFrame:
    return features_from_raw(raw_metrics(df))


def features_from_raw(r: pd.DataFrame) -> pd.DataFrame:
    """Each feature is z-scored across the batch (speed and cadence *within sport*,
    so a swim isn't judged on running pace). Missing data -> 0 (no smell).
    ui/template.html mirrors this function line for line – keep them in sync."""
    hrmax = np.nanmax(r.max_hr.to_numpy(float)) if r.max_hr.notna().any() else 190
    intensity = r.avg_hr / hrmax
    proxy = (r.max_hr - r.avg_hr) / r.avg_hr * 100  # crude stand-in when no streams
    drift = r.drift_pct.fillna(proxy - proxy.mean() + 3)

    f = pd.DataFrame(index=r.index)
    f["speed"] = r.groupby("type").speed_ms.transform(_z)
    f["easy"] = -_z(intensity)
    f["cadence"] = r.groupby("type").cadence.transform(_z)
    f["hr_drift"] = _z(drift)
    f["climb"] = _z(r.climb)
    spiky = _z((r.max_hr - r.avg_hr).astype(float)).fillna(0)
    cv = _z(r.speed_cv).fillna(0)
    f["intervalness"] = 1.5 * r.named + 0.3 * spiky + 0.6 * cv
    return f.fillna(0.0).clip(-3, 3)
