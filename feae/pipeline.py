"""Leakage-free preparation of a stream: every transformer is fitted on the training partition only."""
import time
import warnings

import numpy as np
import pandas as pd


def dfs(mat, feature_defs=None):
    """Deep Feature Synthesis on the PCA output (Featuretools, default primitives).
    The PCA output is a single table without relationships, so the default configuration returns the
    latent coordinates unchanged; the step is kept for completeness and is skipped if Featuretools is unavailable."""
    try:
        import featuretools as ft
        df = pd.DataFrame(mat, columns=[str(i) for i in range(mat.shape[1])]).reset_index(drop=False)
        es = ft.EntitySet(id="d")
        es = es.add_dataframe(dataframe_name="data", dataframe=df, index="index")
        if feature_defs is None:
            fm, defs = ft.dfs(entityset=es, target_dataframe_name="data")
            return fm.values.astype(float), defs
        return ft.calculate_feature_matrix(features=feature_defs, entityset=es).values.astype(float), feature_defs
    except Exception as exc:                                   # noqa: BLE001
        if feature_defs is None:
            warnings.warn(f"Featuretools DFS skipped ({type(exc).__name__}); using the PCA coordinates directly.")
        return np.asarray(mat, float), "identity"


def split_random(y, seed, test_size=0.30):
    from sklearn.model_selection import train_test_split
    return train_test_split(np.arange(len(y)), test_size=test_size, random_state=seed, stratify=y)


def split_novel(y, fam, target, seed, test_size=0.30):
    """Emerging-attack split: the training data contain no flow of family `target`.
    The test stream starts with known traffic only and, from change point h, contains known traffic
    mixed with the unseen family. Returns (train indices, test indices, h)."""
    from sklearn.model_selection import train_test_split
    nov = np.asarray(fam) == target
    idx = np.arange(len(y))
    known = idx[~nov]
    tr, tek = train_test_split(known, test_size=test_size, random_state=seed, stratify=y[known])
    rng = np.random.default_rng(seed)
    tek = tek.copy()
    rng.shuffle(tek)
    h = len(tek) // 2
    p2 = np.concatenate([tek[h:], idx[nov]])
    rng.shuffle(p2)
    return tr, np.concatenate([tek[:h], p2]), h


def prepare(df, label_col, exclude, tr, te, seed, order=None, noise=0.0, poison=0.0, n_trees=300, keep_models=False):
    """Fit imputer, scaler, SMOTE, PCA (95 % variance), DFS and the three base learners on the training rows
    `tr` and return their class probabilities on the test rows `te` (in stream order).

    noise  : std. dev. of Gaussian noise added to the standardised features of the second half of the test stream
    poison : fraction of training labels that are flipped before training
    """
    from imblearn.over_sampling import SMOTE
    from sklearn.decomposition import PCA
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import ComplementNB
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

    t0 = time.time()
    drop = [label_col] + [c for c in exclude if c in df.columns]
    X = df.drop(columns=list(dict.fromkeys(drop))).copy()
    for c in X.select_dtypes(include=["object", "category", "string"]).columns:
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    X = X.replace([np.inf, -np.inf], np.nan)
    y = LabelEncoder().fit_transform(df[label_col].values)
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr].copy(), y[te]
    rng = np.random.default_rng(seed)
    if poison > 0:                                             # flip a fraction of the training labels
        k = rng.choice(len(ytr), int(poison * len(ytr)), replace=False)
        classes = np.unique(y)
        ytr[k] = [rng.choice(classes[classes != v]) for v in ytr[k]]
    imp = SimpleImputer(strategy="mean").fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    Ztr, Zte = sc.transform(imp.transform(Xtr)), sc.transform(imp.transform(Xte))
    if noise > 0:                                              # perturb the second half of the test stream
        h = len(Zte) // 2
        Zte[h:] += rng.normal(0, noise, Zte[h:].shape)
    cnt = np.bincount(ytr)
    k = max(1, min(5, int(cnt[cnt > 0].min()) - 1))
    Ztr, ytr_r = SMOTE(random_state=seed, k_neighbors=k).fit_resample(Ztr, ytr)
    pca = PCA(n_components=0.95, random_state=seed).fit(Ztr)
    Ptr = pca.transform(Ztr)
    Ftr, defs = dfs(Ptr)
    Fte, _ = dfs(pca.transform(Zte), defs)
    Ftr = np.nan_to_num(Ftr, nan=0.0, posinf=0.0, neginf=0.0)
    Fte = np.nan_to_num(Fte, nan=0.0, posinf=0.0, neginf=0.0)
    mm = MinMaxScaler().fit(Ftr)
    lr = LogisticRegression(penalty="l2", C=1.0, solver="saga", max_iter=400).fit(Ftr, ytr_r)
    nb = ComplementNB(alpha=0.1).fit(mm.transform(Ftr), ytr_r)
    et = ExtraTreesClassifier(n_estimators=n_trees, n_jobs=-1, random_state=seed).fit(Ftr, ytr_r)
    out = dict(pA=lr.predict_proba(Fte), pB=nb.predict_proba(np.clip(mm.transform(Fte), 0, None)),
               pC=et.predict_proba(Fte), yte=yte, classes=lr.classes_, pil=Fte[:, :3].copy(),
               order=(None if order is None else np.asarray(order)[te]), n_in=X.shape[1], n_pca=Ptr.shape[1],
               n_dfs=Ftr.shape[1], fit_time=time.time() - t0)
    if keep_models:
        out.update(et=et, Ftr=Ftr, ytr=ytr_r, Fte=Fte)
    return out
