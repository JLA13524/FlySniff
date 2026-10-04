"""Smell an activity.

  features --(joke mapping)--> glomerular odour drive
           --> PN firing --> Kenyon cells (sparse, APL-normalised)
           --> MBON output --> approach / avoid

Two valence routes, like the real fly:
  * innate:  some glomeruli are hard-wired good/bad (CO2, geosmin = bad;
             fruity esters = good). Lateral-horn-ish.
  * learned: the mushroom body. Dopamine (here: kudos) depresses KC->MBON
             synapses for the KCs that were active, shifting the fly's
             opinion of similar-smelling sessions.

None of this is science. It's a real wiring diagram doing a silly job.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .connectome import GLOMERULI, Circuit
from .strava import FEATURES

# feature -> glomerulus. Each row: (glomerulus, real-world odour it detects, caption flavour)
SMELLS = {
    "speed":        ("DM1", "fruity ester",           "smells like a PB"),
    "easy":         ("DM4", "vinegar / fermentation",  "smells like a lovely recovery jog"),
    "cadence":      ("VM2", "ripe fruit",              "smells nicely ripe"),
    "hr_drift":     ("V",   "CO2",                     "smells like panic (CO2)"),
    "climb":        ("DA2", "geosmin – damp soil/mould", "smells like a damp hillside, literally"),
    "intervalness": ("DA1", "cVA – male fly pheromone", "smells like a rival male"),
}
# Innate valence of each glomerulus (rough, from the olfaction literature).
INNATE = {"DM1": 1.0, "DM4": 0.9, "VM2": 0.6, "V": -1.0, "DA2": -1.3, "DA1": -0.8}

# MBON approach(+)/avoid(-) signs, roughly after Aso et al. 2014 (eLife).
# Glutamatergic gamma5/beta'2 outputs -> avoidance; alpha-lobe and gamma1/2 -> approach.
MBON_SIGN = {
    "MBON01": -1, "MBON03": -1, "MBON04": -1, "MBON05": -0.5, "MBON29": -0.5,
    "MBON07": +0.5, "MBON11": +1, "MBON12": +1, "MBON14": +1,
    "MBON15": +0.5, "MBON16": +0.5, "MBON18": +0.5, "MBON19": +0.5,
}

KC_SPARSITY = 0.05  # ~5% of KCs respond to any odour; APL enforces this in vivo


def glom_drive(f: pd.DataFrame) -> np.ndarray:
    """(n_act, n_glom) drive in [0, 1]. Only the positive side of a feature smells."""
    d = np.zeros((len(f), len(GLOMERULI)))
    for feat in FEATURES:
        g = GLOMERULI.index(SMELLS[feat][0])
        d[:, g] = 1 / (1 + np.exp(-2 * (f[feat].to_numpy() - 0.3)))
    return d


class Fly:
    def __init__(self, c: Circuit):
        self.c = c
        n_g = len(GLOMERULI)
        # Glomerulus -> KC input via the real PN->KC synapses.
        g2pn = np.zeros((n_g, len(c.glom_of_pn)))
        g2pn[c.glom_of_pn, np.arange(len(c.glom_of_pn))] = 1
        self.g_kc = g2pn @ c.pn_kc                                  # (n_g, n_kc)
        self.kc_mbon = c.kc_mbon.astype(float).copy()
        self.kc_mbon0 = self.kc_mbon.copy()
        # APL feedback strength per KC: stronger reciprocal APL wiring -> more inhibition.
        apl = np.sqrt(c.kc_apl * c.apl_kc)
        self.apl_gain = apl / apl.mean() if apl.mean() > 0 else np.ones_like(apl)
        self.sign = np.array([MBON_SIGN.get(t.split("_")[0][:6], 0.0) for t in c.mbon_types])
        # Merge left/right copies of the same MBON type so each type gets one vote.
        self.mbon_names = np.array([t.split("_")[0] for t in c.mbon_types])

    # --- forward ---------------------------------------------------------
    def kc(self, drive: np.ndarray) -> np.ndarray:
        x = drive @ self.g_kc                                       # (n_act, n_kc)
        x = x / (self.apl_gain * (1 + 0.2 * x.mean(1, keepdims=True)))
        thr = np.quantile(x, 1 - KC_SPARSITY, axis=1, keepdims=True)
        return np.clip(x - thr, 0, None) / (x.max(1, keepdims=True) - thr + 1e-9)

    def mbon(self, kc: np.ndarray) -> np.ndarray:
        return kc @ self.kc_mbon

    # --- learning (dopamine = kudos) -------------------------------------
    def learn(self, kc: np.ndarray, reward: np.ndarray, lr: float = 0.35):
        """Reward depresses KC->avoid-MBON synapses for active KCs; punishment
        depresses KC->approach-MBON. That's the real MB plasticity rule."""
        for k, r in zip(kc, reward):
            if r == 0:
                continue
            target = (self.sign < 0) if r > 0 else (self.sign > 0)
            dep = lr * abs(r) * np.outer(k, target)
            self.kc_mbon *= np.clip(1 - dep, 0.05, 1)

    # --- whole thing -----------------------------------------------------
    def sniff(self, acts: pd.DataFrame, f: pd.DataFrame, train_on_kudos: bool = True) -> dict:
        drive = glom_drive(f)
        kc = self.kc(drive)
        learned = False
        if train_on_kudos and acts.kudos.notna().sum() >= 4:
            kd = acts.kudos.fillna(acts.kudos.median()).to_numpy(float)
            rew = np.tanh((kd - kd.mean()) / (kd.std() + 1e-9))
            self.learn(kc, rew)
            learned = True

        m = self.mbon(kc)
        m0 = kc @ self.kc_mbon0
        mz = (m - m0.mean(0)) / (m0.std(0) + 1e-9)                   # vs the fly's naive baseline
        mb = np.tanh((mz * self.sign).sum(1) / max(np.abs(self.sign).sum(), 1) * 2)
        innate = np.tanh(1.5 * (drive - 0.3) @ np.array([INNATE[g] for g in GLOMERULI]))
        valence = 0.6 * innate + 0.4 * mb

        out = []
        for i, a in acts.reset_index(drop=True).iterrows():
            # only glomeruli that are actually firing get credit/blame in the caption
            contrib = {feat: float(max(drive[i, GLOMERULI.index(SMELLS[feat][0])] - 0.35, 0)
                                   * INNATE[SMELLS[feat][0]]) for feat in FEATURES}
            out.append({
                "id": a.id, "name": a["name"], "type": a.type,
                "date": None if pd.isna(a.date) else a.date.isoformat(),
                "weekday": None if pd.isna(a.date) else a.date.day_name(),
                "distance_km": _r(a.distance_m / 1000),
                "moving_min": _r(a.moving_s / 60),
                "elev_m": _r(a.elev_m), "avg_hr": _r(a.avg_hr), "kudos": _r(a.kudos),
                "features": {k: round(float(f.iloc[i][k]), 3) for k in FEATURES},
                "glomeruli": {g: round(float(drive[i, j]), 3) for j, g in enumerate(GLOMERULI)},
                "kc_active": int((kc[i] > 0).sum()),
                "kc_pattern": _kc_sample(kc[i]),
                "mbon": {n: round(float(v), 2) for n, v in _merge(self.mbon_names, mz[i]).items()},
                "innate": round(float(innate[i]), 3),
                "learned": round(float(mb[i]), 3),
                "valence": round(float(valence[i]), 3),
                "verdict": _verdict(valence[i]),
                "caption": _caption(a, valence[i], contrib),
                "top_smell": _top(contrib, valence[i]),
            })
        return {"activities": out, "learned_from_kudos": learned, "summary": _summary(out)}


def _r(x, n=1):
    return None if x is None or pd.isna(x) else round(float(x), n)


def _merge(names, vals):
    d = {}
    for n, v in zip(names, vals):
        d.setdefault(n, []).append(v)
    return {n: float(np.mean(v)) for n, v in sorted(d.items())}


def _kc_sample(k: np.ndarray, n: int = 400) -> list[int]:
    """Which of the first n KCs (in a fixed order) fired – drives the UI sparkle."""
    idx = np.linspace(0, len(k) - 1, n).astype(int)
    return [int(i) for i, v in enumerate(k[idx]) if v > 0]


def _top(contrib: dict, v: float) -> str:
    """The smell that most explains the verdict (same sign as the valence)."""
    sgn = 1 if v >= 0 else -1
    return max(contrib, key=lambda k: sgn * contrib[k])


def _verdict(v: float) -> str:
    return ("OBSESSED" if v > .55 else "LOVES" if v > .2 else "into it" if v > .05
            else "indifferent" if v > -.05 else "not a fan" if v > -.2 else "HATES" if v > -.55 else "FLED THE ROOM")


def _caption(a, v, contrib) -> str:
    feat = _top(contrib, v)
    smell = SMELLS[feat][2]
    day = "" if pd.isna(a.date) else a.date.day_name()
    nm = a["name"] or a.type
    if "interval" in nm.lower() or "vo2" in nm.lower():
        subj = f"your {day} intervals" if day else "your intervals"
    else:
        subj = f"“{nm}”"
    phrase = {"OBSESSED": "is obsessed with", "LOVES": "loves", "into it": "is into",
              "not a fan": "is not a fan of", "HATES": "hates"}
    verb = _verdict(v)
    if verb == "indifferent":
        return f"the fly sniffed {subj} and felt nothing."
    if contrib[feat] * v <= 0:  # no smell explains it – the mushroom body just has opinions
        smell = "it's a mushroom body thing"
        return f"the fly {phrase.get(verb, 'is fleeing')} {subj}. no idea why – {smell}."
    if verb == "FLED THE ROOM":
        return f"the fly smelled {subj} and fled the room. it {smell}."
    return f"the fly {phrase[verb]} {subj}. it {smell}."


def _summary(out: list[dict]) -> dict:
    s = sorted(out, key=lambda a: a["valence"])
    by_day: dict[str, list[float]] = {}
    by_type: dict[str, list[float]] = {}
    for a in out:
        if a["weekday"]:
            by_day.setdefault(a["weekday"], []).append(a["valence"])
        by_type.setdefault(a["type"] or "?", []).append(a["valence"])
    avg = lambda d: {k: round(float(np.mean(v)), 3) for k, v in d.items()}
    return {"most_loved": s[-1]["id"] if s else None, "most_hated": s[0]["id"] if s else None,
            "by_weekday": avg(by_day), "by_type": avg(by_type)}
