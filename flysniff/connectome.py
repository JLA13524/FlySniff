"""Pull the fly's olfactory subcircuit out of the MaleCNS connectome.

We only need the smell pathway, not all 166k neurons:

    uniglomerular PNs (by glomerulus)  ->  Kenyon cells  ->  MBONs
                                            ^      |
                                            +-APL--+   (global inhibition)

Weights are real synapse counts from neuPrint. The result is cached to
`circuit.npz` so you only hit the server once.

`demo_circuit()` builds a synthetic circuit with the same shape so the
whole pipeline runs offline (and in CI).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

NEUPRINT = "https://neuprint.janelia.org"

# Glomeruli the encoder drives. See fly.py for what each one "smells".
GLOMERULI = ["DM1", "DM4", "VM2", "V", "DA2", "DA1"]


@dataclass
class Circuit:
    source: str                 # "maleCNS:<dataset>" or "demo"
    glom_of_pn: np.ndarray      # (n_pn,) index into GLOMERULI
    pn_kc: np.ndarray           # (n_pn, n_kc) synapse counts
    kc_mbon: np.ndarray         # (n_kc, n_mbon)
    kc_apl: np.ndarray          # (n_kc,)
    apl_kc: np.ndarray          # (n_kc,)
    kc_types: np.ndarray        # (n_kc,) str
    mbon_types: np.ndarray      # (n_mbon,) str
    pn_types: np.ndarray        # (n_pn,) str

    def save(self, path: str) -> None:
        np.savez_compressed(path, **{k: np.asarray(v) for k, v in self.__dict__.items()})

    @classmethod
    def load(cls, path: str) -> "Circuit":
        z = np.load(path, allow_pickle=False)
        d = {k: z[k] for k in z.files}
        d["source"] = str(d["source"])
        return cls(**d)

    def describe(self) -> dict:
        return {
            "source": self.source,
            "n_pn": int(len(self.pn_types)),
            "n_kc": int(len(self.kc_types)),
            "n_mbon": int(len(self.mbon_types)),
            "pn_kc_synapses": int(self.pn_kc.sum()),
            "kc_mbon_synapses": int(self.kc_mbon.sum()),
        }


# --------------------------------------------------------------------------
# neuPrint (raw HTTP so the only dependency is `requests`)
# --------------------------------------------------------------------------

class NeuPrint:
    def __init__(self, token: str, dataset: str | None = None, server: str = NEUPRINT):
        self.server = server.rstrip("/")
        self.s = requests.Session()
        self.s.headers["Authorization"] = f"Bearer {token}"
        self.dataset = dataset or self._find_male_cns()

    def _find_male_cns(self) -> str:
        r = self.s.get(f"{self.server}/api/dbmeta/datasets", timeout=30)
        r.raise_for_status()
        names = sorted(k for k in r.json() if "male-cns" in k.lower() or "malecns" in k.lower())
        if not names:
            raise SystemExit(
                "No male-cns dataset visible on this neuPrint server. "
                "Pass --dataset explicitly (see https://neuprint.janelia.org)."
            )
        return names[-1]  # newest version string sorts last

    def cypher(self, q: str) -> pd.DataFrame:
        r = self.s.post(f"{self.server}/api/custom/custom",
                        json={"cypher": q, "dataset": self.dataset}, timeout=300)
        r.raise_for_status()
        j = r.json()
        return pd.DataFrame(j["data"], columns=j["columns"])


def fetch_circuit(token: str, dataset: str | None = None) -> Circuit:
    np_ = NeuPrint(token, dataset)
    print(f"neuPrint dataset: {np_.dataset}")

    glom_alt = "|".join(GLOMERULI)
    # Uniglomerular PNs are typed like "DM1_lPN", "V_l2PN", "DA1_lPN".
    pns = np_.cypher(f"""
        MATCH (n:Neuron) WHERE n.type =~ '^({glom_alt})_.*PN.*'
        RETURN n.bodyId AS id, n.type AS type""")
    kcs = np_.cypher("""
        MATCH (n:Neuron) WHERE n.type STARTS WITH 'KC'
        RETURN n.bodyId AS id, n.type AS type""")
    mbons = np_.cypher("""
        MATCH (n:Neuron) WHERE n.type STARTS WITH 'MBON'
        RETURN n.bodyId AS id, n.type AS type""")
    apl = np_.cypher("""
        MATCH (n:Neuron) WHERE n.type = 'APL' RETURN n.bodyId AS id""")
    print(f"PNs {len(pns)}  KCs {len(kcs)}  MBONs {len(mbons)}  APL {len(apl)}")
    if pns.empty or kcs.empty or mbons.empty:
        raise SystemExit("Couldn't find PN/KC/MBON types; the dataset's type naming may differ.")

    def edges(pre_where: str, post_where: str) -> pd.DataFrame:
        return np_.cypher(f"""
            MATCH (a:Neuron)-[w:ConnectsTo]->(b:Neuron)
            WHERE {pre_where} AND {post_where}
            RETURN a.bodyId AS pre, b.bodyId AS post, w.weight AS weight""")

    pn_where = f"a.type =~ '^({glom_alt})_.*PN.*'"
    e_pk = edges(pn_where, "b.type STARTS WITH 'KC'")
    e_km = edges("a.type STARTS WITH 'KC'", "b.type STARTS WITH 'MBON'")
    e_ka = edges("a.type STARTS WITH 'KC'", "b.type = 'APL'")
    e_ak = edges("a.type = 'APL'", "b.type STARTS WITH 'KC'")

    def dense(e, pre_ids, post_ids):
        pi = {b: i for i, b in enumerate(pre_ids)}
        qi = {b: i for i, b in enumerate(post_ids)}
        m = np.zeros((len(pre_ids), len(post_ids)), dtype=np.float32)
        for a, b, w in e.itertuples(index=False):
            if a in pi and b in qi:
                m[pi[a], qi[b]] += w
        return m

    pn_ids, kc_ids, mb_ids, apl_ids = (list(pns.id), list(kcs.id), list(mbons.id), list(apl.id))
    glom = np.array([GLOMERULI.index(t.split("_")[0]) for t in pns.type])
    return Circuit(
        source=f"maleCNS:{np_.dataset}",
        glom_of_pn=glom,
        pn_kc=dense(e_pk, pn_ids, kc_ids),
        kc_mbon=dense(e_km, kc_ids, mb_ids),
        kc_apl=dense(e_ka, kc_ids, apl_ids).sum(1),
        apl_kc=dense(e_ak, apl_ids, kc_ids).sum(0),
        kc_types=np.array(kcs.type, dtype=str),
        mbon_types=np.array(mbons.type, dtype=str),
        pn_types=np.array(pns.type, dtype=str),
    )


# --------------------------------------------------------------------------
# Offline stand-in with the same anatomy-ish shape
# --------------------------------------------------------------------------

DEMO_MBONS = ["MBON01", "MBON03", "MBON04", "MBON05", "MBON07", "MBON11",
              "MBON12", "MBON14", "MBON15", "MBON16", "MBON18", "MBON19"]


def demo_circuit(seed: int = 7) -> Circuit:
    rng = np.random.default_rng(seed)
    pns_per_glom = [3, 3, 3, 2, 2, 3]
    glom = np.repeat(np.arange(len(GLOMERULI)), pns_per_glom)
    n_pn, n_kc, n_mb = len(glom), 2000, len(DEMO_MBONS)

    # Each KC samples ~6 PN inputs (claws), random but weighted by PN count.
    pn_kc = np.zeros((n_pn, n_kc), np.float32)
    for k in range(n_kc):
        claws = rng.choice(n_pn, size=rng.integers(4, 9), replace=True)
        for c in claws:
            pn_kc[c, k] += rng.integers(3, 12)
    kc_types = rng.choice(["KCg-m", "KCab-c", "KCab-s", "KCa'b'-m"], n_kc, p=[.45, .2, .2, .15])
    # KC->MBON: each MBON samples a lobe-ish subset of KCs.
    kc_mbon = (rng.random((n_kc, n_mb)) < 0.25) * rng.integers(1, 6, (n_kc, n_mb))
    return Circuit(
        source="demo",
        glom_of_pn=glom,
        pn_kc=pn_kc,
        kc_mbon=kc_mbon.astype(np.float32),
        kc_apl=rng.integers(2, 10, n_kc).astype(np.float32),
        apl_kc=rng.integers(2, 10, n_kc).astype(np.float32),
        kc_types=kc_types.astype(str),
        mbon_types=np.array(DEMO_MBONS),
        pn_types=np.array([f"{GLOMERULI[g]}_demoPN" for g in glom]),
    )


def get_circuit(cache: str, demo: bool, dataset: str | None) -> Circuit:
    if demo:
        return demo_circuit()
    if os.path.exists(cache):
        print(f"Using cached circuit {cache}")
        return Circuit.load(cache)
    token = os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS") or os.environ.get("NEUPRINT_TOKEN")
    if not token:
        raise SystemExit("Set NEUPRINT_TOKEN (get one from your neuprint.janelia.org account page), "
                         "or run with --demo.")
    c = fetch_circuit(token, dataset)
    c.save(cache)
    print(f"Cached to {cache}")
    return c
