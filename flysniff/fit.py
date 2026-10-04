"""Read .fit / .fit.gz activity files (Garmin, Wahoo, Strava's bulk export).

Stream metrics are defined in `stream_metrics()`; ui/template.html computes the
same numbers in the browser, so keep the two in sync.
"""
from __future__ import annotations

import gzip
import io
import os
from glob import glob

import numpy as np
import pandas as pd

SPORTS = {"running": "Run", "cycling": "Ride", "swimming": "Swim", "walking": "Walk",
          "hiking": "Hike", "training": "Workout", "rowing": "Row", "e_biking": "Ride",
          "generic": "Workout", "fitness_equipment": "Workout", "cross_country_skiing": "Ski"}


def stream_metrics(hr, v, alt) -> dict:
    """drift_pct: Pa:HR decoupling (speed/HR, 1st vs 2nd half of moving time).
    ascent_m: climbing from a 10-sample smoothed altitude trace.
    speed_cv: variability of 30-sample-averaged moving speed (intervals spike it)."""
    hr, v, alt = (np.asarray(x, float) for x in (hr, v, alt))
    out = {"drift_pct": np.nan, "ascent_m": np.nan, "speed_cv": np.nan}
    if len(alt) > 10 and np.isfinite(alt).sum() > 10:
        a = pd.Series(alt).interpolate(limit_direction="both").rolling(10, min_periods=1).mean().to_numpy()
        d = np.diff(a)
        out["ascent_m"] = float(d[d > 0].sum())
    moving = np.isfinite(v) & (v > 0.5)
    if moving.sum() >= 60:
        vm = pd.Series(v[moving]).rolling(30, min_periods=1).mean().to_numpy()
        out["speed_cv"] = float(vm.std() / vm.mean()) if vm.mean() > 0 else np.nan
        ok = moving & np.isfinite(hr) & (hr > 40)
        if ok.sum() >= 120:
            vv, hh = v[ok], hr[ok]
            h = len(vv) // 2
            e1, e2 = vv[:h].mean() / hh[:h].mean(), vv[h:].mean() / hh[h:].mean()
            out["drift_pct"] = float((e1 - e2) / e1 * 100)
    return out


def _open(path: str):
    with open(path, "rb") as fh:
        raw = fh.read()
    if path.endswith(".gz") or raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return io.BytesIO(raw)


def read_fit(path: str) -> dict | None:
    import fitdecode  # optional dependency: pip install fitdecode

    sess, sport_msg, t, hr, v, alt, cad, dist = {}, None, [], [], [], [], [], []
    with fitdecode.FitReader(_open(path), check_crc=fitdecode.CrcCheck.DISABLED) as fr:
        for m in fr:
            if not isinstance(m, fitdecode.FitDataMessage):
                continue
            if m.name == "sport" and m.has_field("sport"):
                sport_msg = sport_msg or m.get_value("sport")
            elif m.name == "session" and not sess:
                sess = {f.name: f.value for f in m.fields}
            elif m.name == "record":
                g = lambda *ks: next((m.get_value(k) for k in ks if m.has_field(k) and m.get_value(k) is not None), None)
                t.append(g("timestamp"))
                hr.append(g("heart_rate"))
                v.append(g("enhanced_speed", "speed"))
                alt.append(g("enhanced_altitude", "altitude"))
                cad.append(g("cadence"))
                dist.append(g("distance"))
    if not t and not sess:
        return None
    f = lambda xs: np.array([np.nan if x is None else float(x) for x in xs])
    hr_, v_, alt_, cad_, dist_ = f(hr), f(v), f(alt), f(cad), f(dist)
    sm = stream_metrics(hr_, v_, alt_)
    ts = [x for x in t if x is not None]
    start = sess.get("start_time") or (ts[0] if ts else None)
    moving_s = sess.get("total_timer_time") or (len(ts) if ts else np.nan)
    distance = sess.get("total_distance") or (np.nanmax(dist_) if np.isfinite(dist_).any() else np.nan)
    raw_sport = sess.get("sport") or sport_msg or "generic"
    sport = SPORTS.get(str(raw_sport).lower(), str(raw_sport).replace("_", " ").title())
    nm = lambda x: float(np.nanmean(x)) if np.isfinite(x).any() else np.nan
    mx = lambda x: float(np.nanmax(x)) if np.isfinite(x).any() else np.nan
    return {
        "id": os.path.basename(path).split(".")[0],
        "name": _name(path, sport, start, distance),
        "type": sport,
        "date": pd.Timestamp(start) if start is not None else pd.NaT,
        "moving_s": moving_s,
        "speed_ms": sess.get("enhanced_avg_speed") or sess.get("avg_speed")
                    or (distance / moving_s if moving_s else nm(v_)),
        "distance_m": distance,
        "elev_m": sess.get("total_ascent") if sess.get("total_ascent") is not None else sm["ascent_m"],
        "avg_hr": sess.get("avg_heart_rate") or nm(hr_),
        "max_hr": sess.get("max_heart_rate") or mx(hr_),
        "cadence": sess.get("avg_cadence") or sess.get("avg_running_cadence") or nm(cad_),
        "kudos": np.nan,
        "drift_pct": sm["drift_pct"],
        "speed_cv": sm["speed_cv"],
    }


def _name(path, sport, start, distance) -> str:
    stem = os.path.basename(path).split(".")[0]
    if not stem.isdigit():                       # a real file name says more than we can
        return stem.replace("_", " ").replace("-", " ")
    when = pd.Timestamp(start).strftime("%a %-d %b") if start is not None else ""
    km = f"{distance / 1000:.1f} km" if distance and np.isfinite(distance) else ""
    return " · ".join(x for x in [sport, when, km] if x)


def load_fit(paths: list[str]) -> pd.DataFrame:
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob(os.path.join(p, "**", "*.fit"), recursive=True))
            files += sorted(glob(os.path.join(p, "**", "*.fit.gz"), recursive=True))
        else:
            files.append(p)
    rows = []
    for fp in files:
        try:
            r = read_fit(fp)
            if r:
                rows.append(r)
        except Exception as e:  # one corrupt file shouldn't sink the batch
            print(f"skipping {fp}: {e}")
    print(f"read {len(rows)} of {len(files)} FIT files")
    return pd.DataFrame(rows)
