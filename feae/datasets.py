"""Dataset loaders. Each loader returns (dataframe, label column, attack-family column)."""
import glob
import os

import pandas as pd

import config as C


def _p(name):
    return os.path.join(C.DATA_DIR, name)


def load_nsl():
    d = pd.concat([pd.read_csv(_p(f), header=None) for f in C.FILES["NSL"]], ignore_index=True)
    d.columns = [f"f{i}" for i in range(41)] + ["attack", "difficulty_level"]
    d["label"] = (d["attack"] != "normal").astype(int)
    return d, "label", "attack"


def load_unsw():
    d = pd.concat([pd.read_csv(_p(f)) for f in C.FILES["UNSW"]], ignore_index=True)
    return d, "label", "attack_cat"


def load_ton():
    d = pd.read_csv(_p(C.FILES["ToN"]))
    d.columns = [c.strip() for c in d.columns]
    return d, "label", "type"


def load_ddos():
    parts = []
    for f in sorted(glob.glob(_p(C.FILES["DDoS"]))):
        p = pd.read_parquet(f)
        p.columns = [c.strip() for c in p.columns]
        parts.append(p)
    d = pd.concat(parts, ignore_index=True)
    lc = [c for c in d.columns if c.lower() == "label"][0]
    d["atk"] = d[lc].astype(str)
    d["y"] = (d["atk"].str.lower() != "benign").astype(int)
    return d.drop(columns=[lc]), "y", "atk"


def load_iot():
    d = pd.read_csv(_p(C.FILES["IoT"]))
    d.columns = [c.strip() for c in d.columns]
    return d, "Label", "Cat"


LOADERS = dict(ToN=load_ton, NSL=load_nsl, UNSW=load_unsw, DDoS=load_ddos, IoT=load_iot)


def load(name):
    """Load a dataset and draw the fixed random sample used in all experiments."""
    d, label, fam = LOADERS[name]()
    d = d.sample(n=min(C.SAMPLE, len(d)), random_state=0).reset_index(drop=True)
    return d, label, fam
