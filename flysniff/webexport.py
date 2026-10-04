"""Compact JSON form of a circuit for the web page (ui/fitfly.js runs it)."""
import numpy as np

from .connectome import GLOMERULI, Circuit
from .fly import Fly


def circuit_json(c: Circuit) -> dict:
    fly = Fly(c)
    km = fly.kc_mbon
    k, m = np.nonzero(km)
    return {
        "source": c.source,
        "n_kc": int(km.shape[0]),
        "glomeruli": GLOMERULI,
        "g_kc": [[int(round(x)) for x in row] for row in fly.g_kc],          # (n_glom, n_kc)
        "apl_gain": [round(float(x), 4) for x in fly.apl_gain],
        "kc_mbon": [k.tolist(), m.tolist(), [int(x) for x in km[k, m]]],     # sparse triplets
        "mbon_names": fly.mbon_names.tolist(),
        "mbon_sign": fly.sign.tolist(),
    }
