"""DIAGNOSTIC ONLY - model performance audit (does not modify models/, configs, or the test protocol).

Stages (all outputs -> results/audit/):
    python scripts/diagnostic_audit.py data      # descriptive dataset / date / duplicate / metadata statistics
    python scripts/diagnostic_audit.py models    # prod-spec fits on 16k train / 4k val; errors, TF-IDF, separability, ceilings
    python scripts/diagnostic_audit.py prepro    # preprocessing variants A-G (16k train / 4k val)
    python scripts/diagnostic_audit.py chrono    # random vs chronological split inside the 20k pool

Selection discipline: every model fit uses ONLY the 16,000-row train part (or, for chrono, the 20,000-row pool)
and is scored ONLY on the 4,000-row validation part of the pool. The 5,000-row official test split is never
scored; it appears only in purely descriptive, label-count tables (``data`` stage, clearly marked).
"""
from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, recall_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

import run_model_improvement as rmi  # only for load_splits(); importing has no side effects
from src.data_loader import load_dataset
from src.model_selection import FeatureSpec, fit_features, transform_features
from src.preprocessing import STANDARD_STOPWORDS, preprocess_series, preprocess_text, tokenize

warnings.filterwarnings("ignore")
OUT = ROOT / "results" / "audit"
OUT.mkdir(parents=True, exist_ok=True)
SEL = json.loads((ROOT / "results" / "selected_config.json").read_text(encoding="utf-8"))
TAX = json.loads((ROOT / "config" / "taxonomy_v1_conservative.json").read_text(encoding="utf-8"))["mapping"]
V1 = {k: v["normalized_category"] for k, v in TAX.items()}
GROUPS = {}
for k, g in V1.items():
    GROUPS.setdefault(g, []).append(k)
VARIANT_GROUPS = {g: m for g, m in GROUPS.items() if len(m) > 1}


def spec_selected() -> FeatureSpec:
    return rmi._spec_from_selected(SEL)


def clf_selected(y):
    return rmi.make_classifier(dict(SEL["classifier"]), y)


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def parse_dates(s):
    return pd.to_datetime(s, format="mixed", errors="coerce")


def metrics(y, p):
    return dict(acc=float(accuracy_score(y, p)), macro_f1=float(f1_score(y, p, average="macro", zero_division=0)),
                macro_rec=float(recall_score(y, p, average="macro", zero_division=0)),
                weighted_f1=float(f1_score(y, p, average="weighted", zero_division=0)))


def collapse(arr):
    return np.array([V1.get(a, a) for a in arr])


def load_all():
    df = load_dataset(ROOT / "data" / "complaints.csv", drop_invalid=True)
    df["date"] = parse_dates(df["Date received"])
    return df


# ======================================================================================================
def stage_data():
    df = load_all()
    sp = rmi.load_splits()
    res = {}
    n = len(df)
    res["rows"] = n
    res["date_parse_failures"] = int(df["date"].isna().sum())
    res["date_min"], res["date_max"] = str(df["date"].min().date()), str(df["date"].max().date())
    res["dup_complaint_ids"] = int(df["Complaint ID"].duplicated().sum())
    txt = df["text"].astype(str)
    res["empty_narratives"] = int((txt.str.strip().str.len() == 0).sum())
    res["raw_exact_dup_rows"] = int(txt.duplicated(keep="first").sum())
    nw = txt.str.split().str.len()
    res["words_quantiles"] = {str(q): float(nw.quantile(q)) for q in (0, .01, .05, .25, .5, .75, .95, .99, 1)}
    res["lt10_words"] = int((nw < 10).sum()); res["lt20_words"] = int((nw < 20).sum())
    res["gt500_words"] = int((nw > 500).sum()); res["gt1000_words"] = int((nw > 1000).sum())
    mask_cnt = txt.str.lower().str.count(r"\bx{2,}\b")
    res["with_redaction"] = int((mask_cnt > 0).sum()); res["mean_redactions"] = float(mask_cnt.mean())
    res["median_words_missing_text_after_clean"] = int((preprocess_series(txt) == "").sum())
    cleaned = preprocess_series(txt)
    dup_c = cleaned.duplicated(keep=False) & (cleaned != "")
    res["cleaned_dup_rows_in_groups"] = int(dup_c.sum())
    g = pd.DataFrame({"c": cleaned[dup_c], "y": df.loc[dup_c, "category"]}).groupby("c")["y"].agg(["nunique", "size", lambda s: tuple(sorted(set(s)))])
    g.columns = ["nuniq", "size", "labels"]
    res["cleaned_dup_groups"] = int(len(g)); res["cleaned_dup_groups_label_conflict"] = int((g.nuniq > 1).sum())
    res["cleaned_dup_conflict_examples"] = [list(x) for x in g[g.nuniq > 1]["labels"].head(10)]
    res["raw_dup_groups_label_conflict_among_variants_only"] = int(sum(1 for x in g[g.nuniq > 1]["labels"] if len({V1.get(l, l) for l in x}) == 1))
    # first six words (templates)
    first6 = txt.str.lower().str.replace(r"[^a-z0-9 ]", " ", regex=True).str.split().str[:6].str.join(" ")
    res["top_templates_first6"] = first6.value_counts().head(8).to_dict()
    # class distribution & splits (test split counts here are DESCRIPTIVE ONLY)
    counts = df["category"].value_counts()
    tab = pd.DataFrame({"total": counts})
    tab["pct"] = (tab.total / n * 100).round(2)
    for nm, y in (("train16k", sp["y_tr"]), ("val4k", sp["y_val"]), ("pool20k", sp["y_pool"]), ("test5k_descriptive", sp["y_test"])):
        tab[nm] = pd.Series(np.asarray(y)).value_counts().reindex(tab.index).fillna(0).astype(int).values
    tab["in_every_split"] = (tab[["train16k", "val4k", "test5k_descriptive"]] > 0).all(axis=1)
    tab.to_csv(OUT / "class_split_table.csv")
    tr = tab["train16k"]
    res["imbalance_ratio_max_over_min"] = float(counts.max() / counts.min())
    res["majority_prop"] = float(counts.max() / n)
    res["min_support"] = int(counts.min())
    res["classes_lt_train"] = {str(t): int((tr < t).sum()) for t in (10, 25, 50, 100, 250)}
    res["n_classes"] = int(len(counts))
    # timeline
    df["year"] = df["date"].dt.year
    ct = pd.crosstab(df["category"], df["year"]); ct.to_csv(OUT / "category_by_year.csv")
    life = df.groupby("category")["date"].agg(["min", "max", "median", "count"])
    life.to_csv(OUT / "category_lifespan.csv")
    ym = df.assign(ym=df["date"].dt.to_period("M")).groupby(["ym", "category"]).size().unstack(fill_value=0)
    ym.to_csv(OUT / "category_by_month.csv")
    # date separability of variant groups (train threshold -> val)
    dtr = df.loc[sp["y_tr"].index, "date"]; dva = df.loc[sp["y_val"].index, "date"]
    vg = {}
    for gname, members in VARIANT_GROUPS.items():
        trm = np.isin(np.asarray(sp["y_tr"]), members); vam = np.isin(np.asarray(sp["y_val"]), members)
        if trm.sum() == 0 or vam.sum() == 0:
            continue
        pred, true = _date_knn(dtr[trm], np.asarray(sp["y_tr"])[trm], dva[vam]), np.asarray(sp["y_val"])[vam]
        sub = df[df["category"].isin(members)]
        overlap = {}
        for m in members:
            overlap[m] = [str(sub[sub.category == m]["date"].min().date()), str(sub[sub.category == m]["date"].max().date()),
                          int((sub.category == m).sum())]
        vg[gname] = dict(n_val=int(vam.sum()), date_only_variant_acc=float((pred == true).mean()),
                         majority_variant_acc=float(pd.Series(true).value_counts(normalize=True).iloc[0]), lifespans=overlap)
    res["variant_group_date_only_accuracy"] = vg
    # metadata informativeness (lookup tables trained on train only, scored on val)
    meta = {}
    def lookup(ftr, fva):
        ytr = np.asarray(sp["y_tr"]); yva = np.asarray(sp["y_val"])
        key = pd.DataFrame({"f": ftr.fillna("NA").astype(str).values, "y": ytr})
        maj = key.groupby("f")["y"].agg(lambda s: s.value_counts().index[0])
        glob = pd.Series(ytr).value_counts().index[0]
        p = pd.Series(fva.fillna("NA").astype(str).values).map(maj).fillna(glob).values
        known = pd.Series(fva.fillna("NA").astype(str).values).isin(maj.index).mean()
        return dict(val_acc=float((p == yva).mean()), val_macro_f1=float(f1_score(yva, p, average="macro", zero_division=0)),
                    coverage=float(known), n_levels_train=int(len(maj)))
    itr, iva = df.loc[sp["y_tr"].index], df.loc[sp["y_val"].index]
    for nm, cols in (("Issue", ["Issue"]), ("Sub-product", ["Sub-product"]), ("Company", ["Company"]), ("State", ["State"]),
                     ("Submitted via", ["Submitted via"]), ("year", ["year"]), ("Issue+Sub-product", ["Issue", "Sub-product"]),
                     ("year+Issue", ["year", "Issue"]), ("year+Issue+Sub-product", ["year", "Issue", "Sub-product"])):
        meta[nm] = lookup(itr[cols].astype(str).agg("|".join, axis=1) if len(cols) > 1 else itr[cols[0]],
                          iva[cols].astype(str).agg("|".join, axis=1) if len(cols) > 1 else iva[cols[0]])
    meta["majority_class_baseline"] = float((np.asarray(sp["y_val"]) == pd.Series(np.asarray(sp["y_tr"])).value_counts().index[0]).mean())
    res["metadata_lookup_baselines_val"] = meta
    res["issue_missing_pct"] = float(df["Issue"].isna().mean() * 100); res["subproduct_missing_pct"] = float(df["Sub-product"].isna().mean() * 100)
    res["n_unique_issue"] = int(df["Issue"].nunique()); res["n_unique_subproduct"] = int(df["Sub-product"].nunique())
    # issue -> product purity (whole data, descriptive)
    ip = pd.crosstab(df["Issue"], df["category"])
    res["issue_purity_weighted"] = float(ip.max(axis=1).sum() / ip.values.sum())
    sub = df.groupby("Issue")["category"].nunique()
    res["issues_mapping_to_single_product_pct"] = float((sub == 1).mean() * 100)
    dump("data_stats.json", res)
    print(json.dumps(res, indent=1, default=str)[:6000])


def _date_knn(dtrain, ytrain, dquery, k=15):
    """Nearest-in-time label vote (uses only dates; used to test 'is the label variant a function of the date?')."""
    t = dtrain.values.astype("datetime64[D]").astype(np.int64); order = np.argsort(t); t = t[order]; y = np.asarray(ytrain)[order]
    out = []
    for q in dquery.values.astype("datetime64[D]").astype(np.int64):
        i = np.searchsorted(t, q); lo, hi = max(0, i - k), min(len(t), i + k)
        idx = np.arange(lo, hi); idx = idx[np.argsort(np.abs(t[idx] - q))[:k]]
        out.append(pd.Series(y[idx]).value_counts().index[0])
    return np.array(out)


# ======================================================================================================
def top_terms_from_coef(model, names, k=10):
    out = {}
    for ci, c in enumerate(model.classes_):
        row = model.coef_[ci]
        out[c] = [names[j] for j in np.argsort(-row)[:k]]
    return out


VEH = r"\b(car|vehicle|auto|truck|suv|lease|leased|dealer|dealership|repo|repossess\w*|toyota|honda|ford|nissan|chevy|bmw)\b"
PREPAID = r"\b(prepaid|pre-paid|reloadable|green dot|netspend|direct express|gift card|debit card)\b"
BANK = r"\b(checking|savings|bank account|overdraft|deposit|atm|debit)\b"
PAYDAY = r"\b(payday|pay day|title loan|cash advance|short term|installment loan|personal loan)\b"
CRED = r"\b(equifax|experian|transunion|credit report|credit bureau|credit score|credit file)\b"
LEX = {"Vehicle loan or lease": VEH, "Consumer Loan": VEH, "Prepaid card": PREPAID, "Credit card or prepaid card": PREPAID,
       "Credit card": PREPAID, "Bank account or service": BANK, "Checking or savings account": BANK,
       "Payday loan": PAYDAY, "Payday loan, title loan, or personal loan": PAYDAY,
       "Credit reporting": CRED, "Credit reporting, credit repair services, or other personal consumer reports": CRED}


def stage_models():
    df = load_all()
    sp = rmi.load_splits()
    ytr, yva = np.asarray(sp["y_tr"]), np.asarray(sp["y_val"])
    clean_tr, clean_va = preprocess_series(sp["X_tr"]), preprocess_series(sp["X_val"])
    spec = spec_selected()
    blocks, Xtr = fit_features(spec, clean_tr, list(sp["X_tr"]))
    Xva = transform_features(spec, blocks, clean_va, list(sp["X_val"]))
    res = {}
    wv, cv = blocks["word"], blocks["char"]
    nw_, nc_ = len(wv.vocabulary_), len(cv.vocabulary_)
    res["features"] = dict(word=nw_, char=nc_, total=int(Xtr.shape[1]), train_docs=int(Xtr.shape[0]), val_docs=int(Xva.shape[0]),
                           nnz_per_doc_total=float(Xtr.nnz / Xtr.shape[0]), nnz_per_doc_word=float(Xtr[:, :nw_].nnz / Xtr.shape[0]),
                           nnz_per_doc_char=float(Xtr[:, nw_:].nnz / Xtr.shape[0]),
                           sparsity=float(1 - Xtr.nnz / (Xtr.shape[0] * Xtr.shape[1])))
    # --- production artifacts vs. search spec (resolve the 237,148 vs 114,493 question) ---
    import joblib
    try:
        pw = joblib.load(ROOT / "models" / "tfidf_vectorizer.joblib"); pc = joblib.load(ROOT / "models" / "char_vectorizer.joblib")
        pm = joblib.load(ROOT / "models" / "complaint_classifier.joblib")
        res["production_artifacts"] = dict(word_vocab=len(pw.vocabulary_), char_vocab=len(pc.vocabulary_), total=len(pw.vocabulary_) + len(pc.vocabulary_),
                                           word_ngram=list(pw.ngram_range), char_ngram=list(pc.ngram_range), char_analyzer=pc.analyzer,
                                           char_max_features=pc.max_features, classifier=type(pm).__name__, C=float(pm.C),
                                           class_weight=str(pm.class_weight), solver=pm.solver, max_iter=int(pm.max_iter), n_classes=len(pm.classes_))
    except Exception as ex:  # pragma: no cover
        res["production_artifacts"] = f"error {ex}"
    # legacy feature count on the 20k pool (TRAIN pool only; no test text touched)
    base_spec = FeatureSpec.make(word=rmi.BASE_WORD, char=rmi.BASE_CHAR)
    cl_pool = preprocess_series(sp["X_pool"])
    bl, Xp = fit_features(base_spec, cl_pool, None)
    res["legacy_spec_features_on_20k_pool"] = int(Xp.shape[1])
    bl2, Xp2 = fit_features(spec, cl_pool, None)
    res["selected_spec_features_on_20k_pool"] = int(Xp2.shape[1])
    del bl, Xp, bl2, Xp2

    # --- fits ---
    lr = clf_selected(ytr).fit(Xtr, ytr)
    p_lr = lr.predict(Xva); pr_lr = lr.predict_proba(Xva)
    res["LR_selected_val"] = metrics(yva, p_lr)
    svc = LinearSVC(C=0.5, class_weight="balanced", max_iter=5000, random_state=42).fit(Xtr, ytr)
    p_svc = svc.predict(Xva)
    res["LinearSVC_C0.5_balanced_same_features_val"] = metrics(yva, p_svc)
    agree = float((p_lr == p_svc).mean())
    res["LR_vs_SVC"] = dict(agreement=agree, both_wrong=float(((p_lr != yva) & (p_svc != yva)).mean()),
                            only_lr_wrong=float(((p_lr != yva) & (p_svc == yva)).mean()),
                            only_svc_wrong=float(((p_lr == yva) & (p_svc != yva)).mean()),
                            oracle_either_right=float(((p_lr == yva) | (p_svc == yva)).mean()))
    cw = {}
    for nm, cwv in (("none", None), ("balanced", "balanced"), ("power0.5", ("power", 0.5))):
        m = rmi.make_classifier(dict(type="LogisticRegression", C=2, cw=cwv, max_iter=1000), ytr).fit(Xtr, ytr)
        p = m.predict(Xva); pp, rr, ff, ss = precision_recall_fscore_support(yva, p, labels=list(lr.classes_), zero_division=0)
        cw[nm] = dict(**metrics(yva, p), per_class=pd.DataFrame(dict(prec=pp, rec=rr, f1=ff), index=lr.classes_).round(4).to_dict("index"))
    res["class_weight_audit"] = cw
    ctr = pd.Series(ytr).value_counts()
    res["minority_lt100_train_summary"] = {nm: dict(mean_recall=float(np.mean([v["rec"] for k, v in d["per_class"].items() if ctr.get(k, 0) < 100])),
                                                     mean_precision=float(np.mean([v["prec"] for k, v in d["per_class"].items() if ctr.get(k, 0) < 100])))
                                           for nm, d in cw.items()}

    # --- per-class table ---
    labels = list(lr.classes_)
    pp, rr, ff, ss = precision_recall_fscore_support(yva, p_lr, labels=labels, zero_division=0)
    cm = pd.crosstab(pd.Series(yva, name="true"), pd.Series(p_lr, name="pred")).reindex(index=labels, columns=labels, fill_value=0)
    cm.to_csv(OUT / "val_confusion_matrix.csv")
    docs_len = clean_tr.str.split().str.len()
    cent = {}; names_w = np.array(wv.get_feature_names_out())
    Xw = Xtr[:, :nw_]
    for c in labels:
        cent[c] = np.asarray(Xw[ytr == c].mean(axis=0)).ravel()
    C = np.vstack([cent[c] for c in labels]); Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    cos = Cn @ Cn.T
    topsets = {c: set(np.argsort(-cent[c])[:500]) for c in labels}
    coef_terms = top_terms_from_coef(lr, np.concatenate([names_w, np.array(cv.get_feature_names_out())]), k=40)
    word_only_terms = {}
    for ci, c in enumerate(labels):
        row = lr.coef_[ci][:nw_]; word_only_terms[c] = [names_w[j] for j in np.argsort(-row)[:10]]
    rows = []
    for i, c in enumerate(labels):
        errs = cm.loc[c].drop(c).sort_values(ascending=False)
        top_alt = "; ".join(f"{k} ({v})" for k, v in errs.head(2).items() if v > 0)
        jac = {o: len(topsets[c] & topsets[o]) / len(topsets[c] | topsets[o]) for o in labels if o != c}
        nn = max(jac, key=jac.get)
        err_total = int(cm.loc[c].sum() - cm.loc[c, c])
        in_group = int(cm.loc[c, [m for m in GROUPS[V1[c]] if m != c]].sum()) if len(GROUPS[V1[c]]) > 1 else 0
        rows.append(dict(category=c, v1_group=V1[c], train_support=int(ctr.get(c, 0)), val_support=int(ss[i]), precision=pp[i], recall=rr[i], f1=ff[i],
                         errors=err_total, errors_to_sibling_variants=in_group, top_alternatives=top_alt,
                         avg_words_per_doc=float(docs_len[ytr == c].mean()) if (ytr == c).any() else np.nan,
                         nearest_vocab_category=nn, nearest_vocab_jaccard_top500=jac[nn],
                         nearest_centroid_cosine=float(np.sort(cos[i])[-2]),
                         top_words="; ".join(word_only_terms[c])))
    pd.DataFrame(rows).round(4).to_csv(OUT / "per_class_separability.csv", index=False)
    pairs = []
    for i, a in enumerate(labels):
        for j in range(i + 1, len(labels)):
            b = labels[j]
            conf = int(cm.loc[a, b] + cm.loc[b, a])
            ta, tb = set(word_only_terms[a]), set(word_only_terms[b])
            pairs.append(dict(a=a, b=b, val_confusions=conf, centroid_cosine=float(cos[i, j]),
                              top500_jaccard=len(topsets[a] & topsets[b]) / len(topsets[a] | topsets[b]),
                              same_v1_group=V1[a] == V1[b], support_a=int(ss[i]), support_b=int(ss[j])))
    pd.DataFrame(pairs).sort_values("val_confusions", ascending=False).round(4).to_csv(OUT / "pair_separability.csv", index=False)
    # --- error decomposition on val (LR selected) ---
    wmat_tr = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True, lowercase=False)
    Wtr = wmat_tr.fit_transform(clean_tr); Wva = wmat_tr.transform(clean_va)
    nn_sim = np.zeros(len(yva)); nn_lab = np.empty(len(yva), dtype=object)
    for s in range(0, len(yva), 1000):
        S = (Wva[s:s + 1000] @ Wtr.T).toarray(); j = S.argmax(axis=1)
        nn_sim[s:s + 1000] = S[np.arange(len(j)), j]; nn_lab[s:s + 1000] = ytr[j]
    maxp = pr_lr.max(axis=1)
    err = p_lr != yva
    grp_t, grp_p = collapse(yva), collapse(p_lr)
    cat = np.array(["correct"] * len(yva), dtype=object)
    variant = err & (grp_t == grp_p)
    rare = err & ~variant & (pd.Series(yva).map(ctr).fillna(0).values < 50)
    neardup = err & ~variant & ~rare & (nn_sim >= 0.8) & (nn_lab != yva)
    low = err & ~variant & ~rare & ~neardup & (maxp < 0.5)
    conf = err & ~variant & ~rare & ~neardup & ~low
    cat[variant] = "A historical-variant (same v1 group)"; cat[rare] = "B rare true class (<50 train)"
    cat[neardup] = "C near-duplicate contradiction (NN sim>=0.8, NN label != truth)"
    cat[low] = "D cross-product, low confidence (maxp<0.5) = ambiguous"; cat[conf] = "E cross-product, confident (residual)"
    ed = pd.Series(cat[err]).value_counts()
    res["error_decomposition_val"] = dict(total_errors=int(err.sum()), val_n=int(len(yva)), counts=ed.to_dict(),
                                          pct_of_errors=(ed / err.sum() * 100).round(2).to_dict(),
                                          pct_of_all_val=(ed / len(yva) * 100).round(2).to_dict())
    # secondary: all-variant including consumer loan <-> payday family (documented overlap but not collapsed in v1)
    fam = {"Consumer Loan", "Payday loan", "Payday loan, title loan, or personal loan"}
    res["consumer_payday_family_errors"] = int((err & np.isin(yva, list(fam)) & np.isin(p_lr, list(fam))).sum())
    res["collapsed_v1_val"] = metrics(grp_t, grp_p)
    # within-group variant accuracy when group correct
    wg = {}
    for g, mem in VARIANT_GROUPS.items():
        m = np.isin(yva, mem) & np.isin(p_lr, mem)
        wg[g] = dict(n_group_correct=int(m.sum()), variant_exact=float((yva[m] == p_lr[m]).mean()) if m.any() else None)
    res["variant_exactness_given_group_correct"] = wg
    # NN agreement by similarity bin, contradictions
    bins = [0, .5, .7, .8, .9, .95, 1.0001]
    b = pd.cut(nn_sim, bins, right=False)
    nb = pd.DataFrame(dict(bin=b, nn_agree=(nn_lab == yva), lr_acc=(p_lr == yva), nn_group_agree=(collapse(nn_lab) == grp_t)))
    res["nn_similarity_bins_val"] = {str(k): dict(n=int(len(v)), nn_label_agree=float(v.nn_agree.mean()), nn_group_agree=float(v.nn_group_agree.mean()),
                                                 lr_acc=float(v.lr_acc.mean())) for k, v in nb.groupby("bin", observed=True)}
    res["val_docs_with_train_neighbour_sim_ge_0.9"] = int((nn_sim >= .9).sum())
    # train-train near-duplicate conflicts (blocked)
    tot = conflict = conflict_var = 0; ctr_pairs = {}
    for s in range(0, Wtr.shape[0], 1500):
        S = (Wtr[s:s + 1500] @ Wtr.T).toarray()
        ii, jj = np.where(S >= 0.9)
        gi = ii + s; keep = gi < jj
        gi, jj = gi[keep], jj[keep]
        tot += len(gi)
        d = ytr[gi] != ytr[jj]; conflict += int(d.sum())
        sameg = collapse(ytr[gi]) == collapse(ytr[jj])
        conflict_var += int((d & sameg).sum())
        for a_, b_ in zip(ytr[gi][d], ytr[jj][d]):
            ctr_pairs[tuple(sorted((a_, b_)))] = ctr_pairs.get(tuple(sorted((a_, b_))), 0) + 1
    res["train_near_dup_pairs_sim_ge_0.9"] = dict(total=tot, label_conflict=conflict, conflict_same_v1_group=conflict_var,
                                                 conflict_cross_group=conflict - conflict_var,
                                                 top_conflict_pairs=[[list(k), v] for k, v in sorted(ctr_pairs.items(), key=lambda x: -x[1])[:8]])
    # lexical evidence for confusions (a-priori lexicons; raw text)
    lex = {}
    raw_va = pd.Series(list(sp["X_val"])).astype(str).str.lower().values
    raw_tr = pd.Series(list(sp["X_tr"])).astype(str).str.lower().values
    def has(rx, arr):
        r = re.compile(rx); return np.array([bool(r.search(t)) for t in arr])
    for pair in [("Vehicle loan or lease", "Consumer Loan"), ("Prepaid card", "Credit card"), ("Credit card or prepaid card", "Credit card"),
                 ("Checking or savings account", "Bank account or service"), ("Payday loan", "Payday loan, title loan, or personal loan"),
                 ("Credit reporting", "Credit reporting, credit repair services, or other personal consumer reports")]:
        a_, b_ = pair
        d = {}
        for c in pair:
            hh = has(LEX[c], raw_tr)
            d[c] = dict(train_n=int((ytr == c).sum()), train_pct_with_lexicon=float(hh[ytr == c].mean() * 100))
        d["val_errors_between"] = int(((yva == a_) & (p_lr == b_)).sum() + ((yva == b_) & (p_lr == a_)).sum())
        m = ((yva == a_) & (p_lr == b_)) | ((yva == b_) & (p_lr == a_))
        d["val_errors_pct_where_true_label_lexicon_absent"] = float(np.mean([not has(LEX[t], [x])[0] for t, x in zip(yva[m], raw_va[m])]) * 100) if m.any() else None
        lex["  vs  ".join(pair)] = d
    res["lexical_evidence"] = lex

    # --- information availability: date / issue / sub-product added to the text (DIAGNOSTIC; not a proposal to change production) ---
    meta = df
    def enc(cols, fn):
        e = OneHotEncoder(handle_unknown="ignore", min_frequency=1)
        a = e.fit_transform(fn(meta.loc[sp["y_tr"].index, cols])); b = e.transform(fn(meta.loc[sp["y_val"].index, cols])); return a, b
    qtr = lambda d: pd.DataFrame({"q": d["date"].dt.to_period("Q").astype(str)})
    dq_tr, dq_va = enc(["date"], qtr)
    info = {}
    iss_tr, iss_va = enc(["Issue", "Sub-product"], lambda d: d.fillna("NA").astype(str))
    combos = {"text+date(quarter)": (dq_tr, dq_va), "text+Issue+Sub-product": (iss_tr, iss_va),
              "text+date+Issue+Sub-product": (sparse.hstack([dq_tr, iss_tr]).tocsr(), sparse.hstack([dq_va, iss_va]).tocsr())}
    for nm, (a_tr, a_va) in combos.items():
        m = rmi.make_classifier(dict(SEL["classifier"]), ytr).fit(sparse.hstack([Xtr, a_tr]).tocsr(), ytr)
        p = m.predict(sparse.hstack([Xva, a_va]).tocsr()); info[nm] = metrics(yva, p)
    res["information_augmented_val_DIAGNOSTIC"] = info
    # date-resolved variants: text decides the group, date decides the variant (kNN in time on TRAIN labels)
    dtr_all = meta.loc[sp["y_tr"].index, "date"]; dva_all = meta.loc[sp["y_val"].index, "date"]
    resolved = p_lr.copy().astype(object)
    for g, mem in VARIANT_GROUPS.items():
        trm = np.isin(ytr, mem); qm = np.isin(p_lr, mem)
        if qm.any() and trm.any():
            resolved[qm] = _date_knn(dtr_all[trm], ytr[trm], dva_all[qm])
    res["text_group_plus_date_variant_val_DIAGNOSTIC"] = metrics(yva, resolved)
    pd.DataFrame(dict(true=yva, pred_lr=p_lr, pred_svc=p_svc, maxp=maxp, nn_sim=nn_sim, nn_label=nn_lab, error_class=cat,
                      date=dva_all.values)).to_csv(OUT / "val_predictions.csv", index=False)
    # top class terms
    pd.DataFrame({"category": labels, "top_coef_terms_word_and_char": ["; ".join(coef_terms[c][:25]) for c in labels]}).to_csv(OUT / "top_terms_per_class.csv", index=False)
    # search-space summary
    mc = pd.read_csv(ROOT / "results" / "model_comparison.csv")
    res["search_summary"] = dict(n=int(len(mc)), val_acc_min=float(mc["Val Accuracy"].min()), val_acc_max=float(mc["Val Accuracy"].max()),
                                 val_macro_f1_min=float(mc["Val Macro F1"].min()), val_macro_f1_max=float(mc["Val Macro F1"].max()),
                                 val_acc_p90=float(mc["Val Accuracy"].quantile(.9)), stages=mc["Stage"].value_counts().to_dict(),
                                 best_by_model=mc.sort_values("Val Macro F1", ascending=False).groupby("Model").head(1)[["Model", "Class Weight", "C", "Val Accuracy", "Val Macro Rec", "Val Macro F1", "Val Weighted F1", "Stage"]].to_dict("records"),
                                 best_by_accuracy=mc.sort_values("Val Accuracy", ascending=False).head(3)[["Model", "Class Weight", "C", "Val Accuracy", "Val Macro F1", "Stage"]].to_dict("records"),
                                 n_unconverged=int((~mc["Converged"].astype(bool)).sum()) if "Converged" in mc else None)
    dump("models_stats.json", res)
    print(json.dumps(res, indent=1, default=str)[:7000])


# ======================================================================================================
NEG = {"no", "not", "nor", "cannot", "can't", "couldn't", "didn't", "doesn't", "don't", "hadn't", "hasn't", "haven't", "isn't", "mustn't",
       "shan't", "shouldn't", "wasn't", "weren't", "won't", "wouldn't", "aren't", "never"}


def _clean_with(text, keep_pat=None, keep_mask=False, neg=False, symbols=None):
    t = text.lower() if isinstance(text, str) else ""
    t = re.sub(r"https?://\S+|www\.\S+", " ", t); t = re.sub(r"\S+@\S+", " ", t)
    if neg:
        t = re.sub(r"n't\b", " not", t); t = t.replace("cannot", "can not")
    for ch, tok in (symbols or {}).items():
        t = t.replace(ch, f" {tok} ")
    t = re.sub(r"[^a-zA-Z0-9\s]", " ", t)
    t = re.sub(r"\bx{2,}\b", " redactedmask " if keep_mask else " ", t)
    return re.sub(r"\s+", " ", t).strip()


def make_variants():
    stop = STANDARD_STOPWORDS
    return {
        "A current": lambda t: preprocess_text(t),
        "B minimal (lowercase+whitespace only; punctuation, XXXX, stopwords kept)": lambda t: re.sub(r"\s+", " ", t.lower()).strip() if isinstance(t, str) else "",
        "C no stopword removal": lambda t: _clean_with(t),
        "D current + ! ? kept as tokens": lambda t: " ".join(w for w in _clean_with(t, symbols={"!": "exclamationmark", "?": "questionmark"}).split() if w not in stop),
        "E current + $ % kept as tokens": lambda t: " ".join(w for w in _clean_with(t, symbols={"$": "dollarsign", "%": "percentsign"}).split() if w not in stop),
        "F current + negation preserved (n't->not, not/no kept)": lambda t: " ".join(w for w in _clean_with(t, neg=True).split() if (w not in stop) or w in {"no", "not", "nor", "never"}),
        "G current + redaction mask kept as token": lambda t: " ".join(w for w in _clean_with(t, keep_mask=True).split() if w not in stop),
    }


def stage_prepro():
    sp = rmi.load_splits()
    ytr, yva = np.asarray(sp["y_tr"]), np.asarray(sp["y_val"])
    spec = spec_selected()
    pre_file = OUT / "preprocessing_variants.json"
    res = json.loads(pre_file.read_text(encoding="utf-8")) if pre_file.exists() else {}
    raw_tr, raw_va = list(sp["X_tr"]), list(sp["X_val"])
    # negation prevalence (raw)
    s = pd.Series(raw_tr).astype(str).str.lower()
    res["train_docs_with_not_or_nt"] = float(s.str.contains(r"\bnot\b|n't|\bno\b|\bnever\b").mean() * 100)
    res["current_pipeline_turns_nt_into_tokens"] = preprocess_text("I didn't receive it and can't pay. No refund, not fair.")
    for nm, fn in make_variants().items():
        if nm in res:
            print(f"Skipping already evaluated {nm}", flush=True)
            continue
        ctr = pd.Series(raw_tr).map(fn); cva = pd.Series(raw_va).map(fn)
        blocks, Xtr = fit_features(spec, ctr, raw_tr); Xva = transform_features(spec, blocks, cva, raw_va)
        m = clf_selected(ytr).fit(Xtr, ytr); p = m.predict(Xva)
        r = metrics(yva, p); r["features"] = int(Xtr.shape[1]); r["val_group_acc_v1"] = float((collapse(yva) == collapse(p)).mean())
        res[nm] = r; print(nm, r, flush=True)
        dump("preprocessing_variants.json", res)


# ======================================================================================================
def stage_chrono():
    df = load_all(); sp = rmi.load_splits(); spec = spec_selected(); res = {}
    pool_idx = sp["y_pool"].index
    pool = df.loc[pool_idx].copy()
    ypool = pool["category"].values
    cleaned = preprocess_series(pool["text"]); cleaned.index = pool.index
    def run(tr_idx, va_idx, tag):
        ytr_, yva_ = df.loc[tr_idx, "category"].values, df.loc[va_idx, "category"].values
        b, Xt = fit_features(spec, cleaned.loc[tr_idx], None); Xv = transform_features(spec, b, cleaned.loc[va_idx], None)
        m = rmi.make_classifier(dict(SEL["classifier"]), ytr_).fit(Xt, ytr_); p = m.predict(Xv)
        seen = np.isin(yva_, np.unique(ytr_))
        r = dict(metrics(yva_, p), n_train=int(len(ytr_)), n_val=int(len(yva_)), val_group_acc_v1=float((collapse(yva_) == collapse(p)).mean()),
                 val_rows_with_label_unseen_in_train=int((~seen).sum()), acc_on_seen_labels=float((p[seen] == yva_[seen]).mean()),
                 train_date_range=[str(df.loc[tr_idx, "date"].min().date()), str(df.loc[tr_idx, "date"].max().date())],
                 val_date_range=[str(df.loc[va_idx, "date"].min().date()), str(df.loc[va_idx, "date"].max().date())],
                 val_label_counts=pd.Series(yva_).value_counts().to_dict(), train_label_counts=pd.Series(ytr_).value_counts().to_dict())
        res[tag] = r; print(tag, {k: v for k, v in r.items() if "counts" not in k}, flush=True); dump("chrono_diagnostic.json", res)
    # A. the official random 16k/4k split of the pool
    run(sp["y_tr"].index, sp["y_val"].index, "A random stratified 16k/4k (official protocol, pool only)")
    # B. chronological: oldest 16k train, newest 4k val
    order = pool.sort_values("date", kind="mergesort").index
    run(order[:16000], order[16000:], "B chronological oldest16k->newest4k")
    # C. reverse chronological sanity: newest 16k train, oldest 4k val
    run(order[4000:], order[:4000], "C reverse chronological newest16k->oldest4k")
    # D. random split among rows with 'modern' (post-restructure) or stable labels only: removes legacy-label coexistence (diagnostic)
    legacy = {"Credit reporting", "Credit card", "Prepaid card", "Bank account or service", "Money transfers", "Virtual currency", "Payday loan", "Other financial service", "Consumer Loan"}
    mod = pool[~pool["category"].isin(legacy)].index
    from sklearn.model_selection import train_test_split
    tr_i, va_i = train_test_split(mod, test_size=0.2, random_state=42, stratify=df.loc[mod, "category"].values)
    run(tr_i, va_i, "D random split, legacy-label rows removed (DIAGNOSTIC ONLY, changes label set)")
    # E. same, but chrono within modern-label rows
    mo = df.loc[mod].sort_values("date", kind="mergesort").index; k = int(len(mo) * .8)
    run(mo[:k], mo[k:], "E chronological, legacy-label rows removed (DIAGNOSTIC ONLY)")


if __name__ == "__main__":
    stages = {"data": stage_data, "models": stage_models, "prepro": stage_prepro, "chrono": stage_chrono}
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    if arg == "all":
        for s, fn in stages.items():
            print(f"=== Running stage {s} ===", flush=True)
            fn()
    elif arg in stages:
        stages[arg]()
    else:
        print("Usage: python scripts/diagnostic_audit.py [data|models|prepro|chrono|all]")
        sys.exit(0 if arg in ("-h", "--help", "help") else 1)
