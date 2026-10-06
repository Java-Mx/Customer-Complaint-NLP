"""Staged, validation-only model-improvement search + one-shot final test evaluation.

Usage
-----
    python scripts/run_model_improvement.py search   # stages 1-6 on 16k train / 4k val ONLY
    python scripts/run_model_improvement.py final    # retrain selected cfg on 20k pool, evaluate ONCE on 5k test

Protocol (identical to scripts/run_experiments.py): stratified 80/20 split (random_state=42) of the
25,000 CFPB records -> 20,000 training pool / 5,000 untouched test set; the pool is split again 80/20
(random_state=42) -> 16,000 train / 4,000 validation. The ``search`` command never touches the test set.
The ``final`` command refuses to run twice (guard file) so the test set is evaluated exactly once.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.classification import train_test_split_data
from src.data_loader import load_dataset
from src.model_selection import (
    RANDOM_STATE,
    FeatureSpec,
    fit_features,
    power_class_weights,
    rank_candidates,
    transform_features,
    validation_metrics,
)
from src.preprocessing import preprocess_series

RESULTS = ROOT_DIR / "results"
COMPARISON_CSV = RESULTS / "model_comparison.csv"
SELECTED_JSON = RESULTS / "selected_config.json"
FINAL_JSON = RESULTS / "model_improvement_test_metrics.json"
LEGACY_CSV = RESULTS / "model_comparison_legacy.csv"
N_JOBS = 4

BASE_WORD = dict(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True, norm="l2")
BASE_CHAR = dict(ngram_range=(3, 5), min_df=5, max_df=0.95, sublinear_tf=True, analyzer="char_wb")


# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------
def load_splits() -> Dict[str, Any]:
    df = load_dataset(ROOT_DIR / "data" / "complaints.csv", drop_invalid=True)
    X_pool, X_test, y_pool, y_test = train_test_split_data(df, test_size=0.20, random_state=RANDOM_STATE, stratify=True)
    df_pool = pd.DataFrame({"text": X_pool, "category": y_pool})
    X_tr, X_val, y_tr, y_val = train_test_split_data(df_pool, test_size=0.20, random_state=RANDOM_STATE, stratify=True)
    return dict(X_pool=X_pool, y_pool=y_pool, X_test=X_test, y_test=y_test,
                X_tr=X_tr, y_tr=y_tr, X_val=X_val, y_val=y_val)


# ----------------------------------------------------------------------------------------------
# classifiers
# ----------------------------------------------------------------------------------------------
def resolve_class_weight(cw, y_train):
    """cw: None | 'balanced' | ('power', p) -> value for sklearn (computed from y_train only)."""
    if isinstance(cw, (tuple, list)) and cw and cw[0] == "power":
        return power_class_weights(y_train, float(cw[1]))
    return cw


def cw_label(cw) -> str:
    if isinstance(cw, (tuple, list)):
        return f"power={cw[1]}"
    return str(cw)


def make_classifier(clf: Dict[str, Any], y_train):
    t = clf["type"]
    if t == "LogisticRegression":
        return LogisticRegression(C=clf["C"], class_weight=resolve_class_weight(clf.get("cw"), y_train),
                                  solver="lbfgs", max_iter=clf.get("max_iter", 500), random_state=RANDOM_STATE)
    if t == "LinearSVC":
        return LinearSVC(C=clf["C"], class_weight=resolve_class_weight(clf.get("cw"), y_train),
                         max_iter=clf.get("max_iter", 5000), random_state=RANDOM_STATE)
    if t == "ComplementNB":
        return ComplementNB(alpha=clf["alpha"])
    if t == "MultinomialNB":
        return MultinomialNB(alpha=clf["alpha"])
    raise ValueError(t)


def clf_params(clf: Dict[str, Any]) -> str:
    return json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in clf.items() if k != "type"}, sort_keys=True)


def fit_eval(stage: str, spec: FeatureSpec, X_tr, X_val, y_tr, y_val, clf: Dict[str, Any], t_feat: float = 0.0) -> Dict[str, Any]:
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = make_classifier(clf, y_tr)
        model.fit(X_tr, np.asarray(y_tr))
        pred = model.predict(X_val)
    fit_time = time.time() - t0
    converged = True
    if hasattr(model, "n_iter_"):
        n_it = int(np.max(model.n_iter_))
        converged = n_it < clf.get("max_iter", 500 if clf["type"] == "LogisticRegression" else 5000)
    row = {
        "Model": clf["type"],
        "Features": spec.describe(),
        "Dimensions": int(X_tr.shape[1]),
        "Class Weight": cw_label(clf.get("cw")) if "cw" in clf else "n/a",
        "C": clf.get("C", clf.get("alpha", "")),
        **{k: round(v, 4) for k, v in validation_metrics(y_val, pred).items()},
        "Fit Time (s)": round(fit_time + t_feat, 2),
        "Stage": stage,
        "Classifier Params": clf_params(clf),
        "Converged": converged,
        "Random State": RANDOM_STATE,
        "Train Size": int(X_tr.shape[0]),
        "Val Size": int(X_val.shape[0]),
    }
    return row


def feature_job(stage: str, spec: FeatureSpec, clean_tr, clean_val, raw_tr, raw_val, y_tr, y_val, clfs) -> List[Dict[str, Any]]:
    t0 = time.time()
    blocks, X_tr = fit_features(spec, clean_tr, raw_tr)          # fit on TRAIN only
    X_val = transform_features(spec, blocks, clean_val, raw_val)  # transform val
    t_feat = time.time() - t0
    return [fit_eval(stage, spec, X_tr, X_val, y_tr, y_val, c, t_feat) for c in clfs]


def clf_job(stage: str, spec: FeatureSpec, X_tr, X_val, y_tr, y_val, clf) -> Dict[str, Any]:
    return fit_eval(stage, spec, X_tr, X_val, y_tr, y_val, clf)


# ----------------------------------------------------------------------------------------------
# search
# ----------------------------------------------------------------------------------------------
SCREEN_CLFS = [
    {"type": "LinearSVC", "C": 0.5, "cw": "balanced"},
    {"type": "LogisticRegression", "C": 1.0, "cw": "balanced", "max_iter": 500},
]


def best_row(rows: List[Dict[str, Any]], stage: Optional[str] = None) -> Dict[str, Any]:
    df = pd.DataFrame(rows)
    if stage:
        df = df[df["Stage"] == stage]
    return rank_candidates(df).iloc[0].to_dict()


def save(rows, path=COMPARISON_CSV):
    df = rank_candidates(pd.DataFrame(rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def spec_from_row(row: Dict[str, Any], specs: Dict[str, FeatureSpec]) -> FeatureSpec:
    return specs[row["Features"]]


def run_search():
    t_all = time.time()
    sp = load_splits()
    print(f"split sizes: train={len(sp['X_tr'])} val={len(sp['X_val'])} (pool={len(sp['X_pool'])}, test={len(sp['X_test'])} NOT USED)", flush=True)
    clean_tr, clean_val = preprocess_series(sp["X_tr"]), preprocess_series(sp["X_val"])
    raw_tr, raw_val = list(sp["X_tr"]), list(sp["X_val"])
    y_tr, y_val = np.asarray(sp["y_tr"]), np.asarray(sp["y_val"])
    rows: List[Dict[str, Any]] = []
    specs: Dict[str, FeatureSpec] = {}

    if COMPARISON_CSV.exists():
        try:
            existing_df = pd.read_csv(COMPARISON_CSV)
            rows = existing_df.to_dict(orient="records")
            print(f"Loaded {len(rows)} existing candidate evaluations from {COMPARISON_CSV}", flush=True)
        except Exception as ex:
            print(f"Could not load existing {COMPARISON_CSV}: {ex}", flush=True)

    completed_stages = {str(r.get("Stage", "")) for r in rows}
    if SELECTED_JSON.exists() and {"S0 reference", "S1 word", "S2 char", "S2 word+char", "S3 LR", "S4 SVC", "S4 NB", "S5 class-weights"}.issubset(completed_stages):
        print(f"All search stages evaluated ({len(rows)} candidates) and {SELECTED_JSON.name} exists.", flush=True)
        return

    def run_specs(stage, spec_list, clfs=SCREEN_CLFS):
        for s in spec_list:
            specs[s.describe()] = s
        out = Parallel(n_jobs=N_JOBS, verbose=0)(
            delayed(feature_job)(stage, s, clean_tr, clean_val, raw_tr, raw_val, y_tr, y_val, clfs) for s in spec_list)
        new = [r for chunk in out for r in chunk]
        rows.extend(new)
        save(rows)
        for r in sorted(new, key=lambda r: -r["Val Macro F1"]):
            print(f"  [{stage}] {r['Model']:18} {r['Features'][:90]:90} dims={r['Dimensions']:>7} "
                  f"MF1={r['Val Macro F1']:.4f} Acc={r['Val Accuracy']:.4f}", flush=True)

    # ---- Stage 0: reference = the current production configuration, converged LR ----
    ref_spec = FeatureSpec.make(word=BASE_WORD, char=BASE_CHAR)
    specs[ref_spec.describe()] = ref_spec
    if "S0 reference" not in completed_stages:
        print("\n== Stage 0: reference (current config, max_iter=500) ==", flush=True)
        run_specs("S0 reference", [ref_spec])
    else:
        print("Stage 0 reference already evaluated. Using existing results.", flush=True)

    # ---- Stage 1: word TF-IDF ----
    W = lambda **kw: FeatureSpec.make(word={**BASE_WORD, **kw})
    word_specs = [
        W(), W(ngram_range=(1, 1)), W(ngram_range=(1, 3)),
        W(min_df=3), W(min_df=5), W(max_df=0.5), W(max_df=0.8), W(sublinear_tf=False),
        W(max_features=50000), W(max_features=100000), W(norm="l1"),
        W(ngram_range=(1, 3), min_df=3), W(ngram_range=(1, 3), min_df=5),
    ]
    for s in word_specs:
        specs[s.describe()] = s
    if "S1 word" not in completed_stages:
        print("\n== Stage 1: word TF-IDF configurations ==", flush=True)
        run_specs("S1 word", word_specs)
    else:
        print("Stage 1 word already evaluated. Using existing results.", flush=True)

    best_word_row = best_row(rows, "S1 word")
    best_word = specs[best_word_row["Features"]]
    print("  best word spec:", best_word.describe(), flush=True)

    # ---- Stage 2: char TF-IDF, then word+char ----
    print("\n== Stage 2: char TF-IDF (char vs char_wb) and word+char ==", flush=True)
    char_specs = []
    for an in ("char_wb", "char"):
        for ng in ((2, 5), (3, 5), (3, 6), (4, 6)):
            p = {**BASE_CHAR, "ngram_range": ng, "analyzer": an}
            if an == "char":
                p["max_features"] = 100000
            char_specs.append(FeatureSpec.make(char=p))
    for s in char_specs:
        specs[s.describe()] = s
    if "S2 char" not in completed_stages:
        run_specs("S2 char", char_specs)
    else:
        print("Stage 2 char already evaluated. Using existing results.", flush=True)

    char_ranked = rank_candidates(pd.DataFrame([r for r in rows if r["Stage"] == "S2 char"]))
    top_chars = []
    for feat in char_ranked["Features"]:
        if feat not in top_chars:
            top_chars.append(feat)
        if len(top_chars) == 3:
            break
    combo_specs = []
    for feat in top_chars:
        cs = specs[feat]
        combo_specs.append(FeatureSpec.make(word=dict(best_word.word), char=dict(cs.char)))
    # also the best word config with the *current* char block
    combo_specs.append(FeatureSpec.make(word=dict(best_word.word), char=BASE_CHAR))
    combo_specs = list({s.describe(): s for s in combo_specs}.values())
    for s in combo_specs:
        specs[s.describe()] = s
    if "S2 word+char" not in completed_stages:
        run_specs("S2 word+char", combo_specs)
    else:
        print("Stage 2 word+char already evaluated. Using existing results.", flush=True)

    cand = best_row([r for r in rows if r["Stage"] in ("S1 word", "S2 char", "S2 word+char", "S0 reference")])
    best_spec = specs[cand["Features"]]
    print("  best feature spec after stage 2:", best_spec.describe(), "| clf", cand["Model"], flush=True)

    # materialise best features once for classifier stages
    blocks, X_tr = fit_features(best_spec, clean_tr, raw_tr)
    X_val = transform_features(best_spec, blocks, clean_val, raw_val)
    print(f"  best features materialised: {X_tr.shape}", flush=True)

    def run_clfs(stage, spec, Xtr, Xva, clf_list):
        out = Parallel(n_jobs=N_JOBS, verbose=0)(delayed(clf_job)(stage, spec, Xtr, Xva, y_tr, y_val, c) for c in clf_list)
        rows.extend(out)
        save(rows)
        for r in sorted(out, key=lambda r: -r["Val Macro F1"])[:6]:
            print(f"  [{stage}] {r['Model']:18} cw={r['Class Weight']:10} C={r['C']:<6} MF1={r['Val Macro F1']:.4f} "
                  f"Acc={r['Val Accuracy']:.4f} conv={r['Converged']}", flush=True)

    # ---- Stage 3: Logistic Regression grid ----
    print("\n== Stage 3: Logistic Regression C x class_weight ==", flush=True)
    lr_grid = [{"type": "LogisticRegression", "C": c, "cw": cw, "max_iter": 500}
               for c in (0.1, 0.25, 0.5, 1, 2, 4, 8) for cw in (None, "balanced")]
    if "S3 LR" not in completed_stages:
        run_clfs("S3 LR", best_spec, X_tr, X_val, lr_grid)
    else:
        print("Stage 3 LR already evaluated. Using existing results.", flush=True)

    # ---- Stage 4: LinearSVC + NB ----
    print("\n== Stage 4: LinearSVC and Naive Bayes ==", flush=True)
    svc_grid = [{"type": "LinearSVC", "C": c, "cw": cw}
                for c in (0.05, 0.1, 0.25, 0.5, 1, 2) for cw in (None, "balanced")]
    if "S4 SVC" not in completed_stages:
        run_clfs("S4 SVC", best_spec, X_tr, X_val, svc_grid)
    else:
        print("Stage 4 SVC already evaluated. Using existing results.", flush=True)

    nb_grid = [{"type": t, "alpha": a} for t in ("ComplementNB", "MultinomialNB") for a in (0.01, 0.03, 0.1, 0.3, 1.0)]
    if "S4 NB" not in completed_stages:
        run_clfs("S4 NB", best_spec, X_tr, X_val, nb_grid)
    else:
        print("Stage 4 NB already evaluated. Using existing results.", flush=True)

    # ---- Stage 5: train-only custom class weights (support-based; no val/test info) ----
    print("\n== Stage 5: softened / strengthened support-based class weights ==", flush=True)
    top_lr = best_row([r for r in rows if r["Model"] == "LogisticRegression" and "S3" in r["Stage"]])
    top_svc = best_row([r for r in rows if r["Model"] == "LinearSVC" and "S4" in r["Stage"]])
    custom = []
    for p in (0.25, 0.5, 0.75, 1.25):
        custom.append({"type": "LogisticRegression", "C": float(top_lr["C"]), "cw": ("power", p), "max_iter": 500})
        custom.append({"type": "LinearSVC", "C": float(top_svc["C"]), "cw": ("power", p)})
    if "S5 class-weights" not in completed_stages:
        run_clfs("S5 class-weights", best_spec, X_tr, X_val, custom)
    else:
        print("Stage 5 class-weights already evaluated. Using existing results.", flush=True)

    # ---- Stage 6: stateless text statistics (generic, error-driven) ----
    print("\n== Stage 6: + text statistics (length, redaction-mask count) ==", flush=True)
    stat_specs = [FeatureSpec.make(word=dict(best_spec.word) if best_spec.word else None,
                                   char=dict(best_spec.char) if best_spec.char else None, text_stats=True)]
    for s in stat_specs:
        specs[s.describe()] = s
    top_overall = best_row([r for r in rows if r["Stage"] in ("S3 LR", "S4 SVC", "S4 NB", "S5 class-weights")])
    clf_best = json.loads(top_overall["Classifier Params"])
    clf_best["type"] = top_overall["Model"]
    if "cw" in clf_best and isinstance(clf_best["cw"], list):
        clf_best["cw"] = tuple(clf_best["cw"])
    stat_clfs = [clf_best] + [c for c in SCREEN_CLFS if c["type"] != clf_best["type"]]
    if "S6 text-stats" not in completed_stages:
        run_specs("S6 text-stats", stat_specs, stat_clfs)
    else:
        print("Stage 6 text-stats already evaluated. Using existing results.", flush=True)


    # ---- selection ----
    final_df = rank_candidates(pd.DataFrame(rows))
    winner = final_df.iloc[0].to_dict()
    winner_spec = specs[winner["Features"]]
    clf_cfg = json.loads(winner["Classifier Params"])
    clf_cfg["type"] = winner["Model"]
    selected = {
        "model": winner["Model"],
        "classifier": clf_cfg,
        "word": dict(winner_spec.word) if winner_spec.word else None,
        "char": dict(winner_spec.char) if winner_spec.char else None,
        "text_stats": winner_spec.text_stats,
        "stage": winner["Stage"],
        "validation": {k: float(winner[k]) for k in final_df.columns if k.startswith("Val ")},
        "n_candidates_evaluated": int(len(final_df)),
        "selection_order": ["Val Macro F1", "Val Macro Rec", "Val Accuracy", "Val Weighted F1"],
    }
    # JSON can't hold tuples natively -> lists; ngram ranges restored on load
    SELECTED_JSON.write_text(json.dumps(selected, indent=2, default=list), encoding="utf-8")
    print("\n== SELECTED (validation) ==")
    print(json.dumps(selected, indent=2, default=list))
    ref = [r for r in rows if r["Stage"] == "S0 reference" and r["Model"] == "LogisticRegression"][0]
    print(f"reference (current config, converged LR) val: MF1={ref['Val Macro F1']} Acc={ref['Val Accuracy']}")
    print(f"total candidates: {len(final_df)}  elapsed {time.time()-t_all:.0f}s")


# ----------------------------------------------------------------------------------------------
# final (test set touched ONCE here)
# ----------------------------------------------------------------------------------------------
CURRENT_METRICS = dict(accuracy=0.6956, macro_precision=0.4967, macro_recall=0.5197, macro_f1=0.5056,
                       weighted_precision=0.7003, weighted_recall=0.6956, weighted_f1=0.6955)


def _spec_from_selected(sel: Dict[str, Any]) -> FeatureSpec:
    def fix(d):
        if d is None:
            return None
        d = dict(d)
        if "ngram_range" in d:
            d["ngram_range"] = tuple(d["ngram_range"])
        return d
    return FeatureSpec.make(word=fix(sel["word"]), char=fix(sel["char"]), text_stats=sel["text_stats"])


def run_final(force: bool = False):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
    if FINAL_JSON.exists() and not force:
        raise SystemExit(f"{FINAL_JSON} exists: the test set has already been evaluated once. Refusing to re-evaluate.")
    sel = json.loads(SELECTED_JSON.read_text(encoding="utf-8"))
    spec = _spec_from_selected(sel)
    clf_cfg = dict(sel["classifier"])
    if isinstance(clf_cfg.get("cw"), list):
        clf_cfg["cw"] = tuple(clf_cfg["cw"])
    sp = load_splits()
    clean_pool, clean_test = preprocess_series(sp["X_pool"]), preprocess_series(sp["X_test"])
    y_pool, y_test = np.asarray(sp["y_pool"]), np.asarray(sp["y_test"])
    t0 = time.time()
    blocks, X_pool = fit_features(spec, clean_pool, list(sp["X_pool"]))   # fit on 20k pool only
    X_test = transform_features(spec, blocks, clean_test, list(sp["X_test"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = make_classifier(clf_cfg, y_pool)
        model.fit(X_pool, y_pool)
    print(f"retrained on {X_pool.shape} in {time.time()-t0:.1f}s", flush=True)
    pred = model.predict(X_test)                                          # the single test evaluation

    labels = list(model.classes_)
    p, r, f, s = precision_recall_fscore_support(y_test, pred, labels=labels, zero_division=0)
    def agg(avg):
        pp, rr, ff, _ = precision_recall_fscore_support(y_test, pred, average=avg, zero_division=0)
        return float(pp), float(rr), float(ff)
    mp, mr, mf = agg("macro"); wp, wr, wf = agg("weighted")
    new = dict(accuracy=float(accuracy_score(y_test, pred)), macro_precision=mp, macro_recall=mr, macro_f1=mf,
               weighted_precision=wp, weighted_recall=wr, weighted_f1=wf)

    # exact current-model figures (re-score the stored production model; reference only, not used for selection)
    cur_clf = joblib.load(ROOT_DIR / "models" / "complaint_classifier.joblib")
    cur_w = joblib.load(ROOT_DIR / "models" / "tfidf_vectorizer.joblib")
    cur_c = joblib.load(ROOT_DIR / "models" / "char_vectorizer.joblib")
    from src.vectorization import transform_word_char
    cur_pred = cur_clf.predict(transform_word_char(cur_w, cur_c, clean_test))
    cp, cr, cf, _ = precision_recall_fscore_support(y_test, cur_pred, labels=labels, zero_division=0)
    cm_, cr_, cf_ = [float(x) for x in precision_recall_fscore_support(y_test, cur_pred, average="macro", zero_division=0)[:3]]
    cw_, cwr_, cwf_ = [float(x) for x in precision_recall_fscore_support(y_test, cur_pred, average="weighted", zero_division=0)[:3]]
    cur = dict(accuracy=float(accuracy_score(y_test, cur_pred)), macro_precision=cm_, macro_recall=cr_, macro_f1=cf_,
               weighted_precision=cw_, weighted_recall=cwr_, weighted_f1=cwf_)
    diff_pp = {k: round((new[k] - cur[k]) * 100, 2) for k in new}

    # decision rule (fixed in advance): replace only if Macro F1 improves AND accuracy does not drop
    improved = bool(new["macro_f1"] > cur["macro_f1"] and new["accuracy"] >= cur["accuracy"])

    per_cat = pd.DataFrame({"category": labels, "support": s, "precision": p, "recall": r, "f1": f,
                            "current_precision": cp, "current_recall": cr, "current_f1": cf})
    per_cat["f1_diff_pp"] = ((per_cat["f1"] - per_cat["current_f1"]) * 100).round(2)
    per_cat.sort_values("support", ascending=False).to_csv(RESULTS / "model_improvement_per_category.csv", index=False)
    pd.DataFrame(confusion_matrix(y_test, pred, labels=labels), index=labels, columns=labels).to_csv(
        RESULTS / "model_improvement_confusion_matrix.csv")

    out = dict(selected_config=sel, new_test=new, current_test=cur, improvement_pp=diff_pp,
               recorded_current_metrics=CURRENT_METRICS, improved=improved,
               replacement_rule="replace models/ only if test Macro F1 > current AND test accuracy >= current",
               test_samples=int(len(y_test)), training_samples=int(len(y_pool)), n_features=int(X_pool.shape[1]),
               models_replaced=False)
    if improved:
        models_dir = ROOT_DIR / "models"
        joblib.dump(model, models_dir / "complaint_classifier.joblib")
        if spec.word is not None:
            joblib.dump(blocks["word"], models_dir / "tfidf_vectorizer.joblib")
        char_path = models_dir / "char_vectorizer.joblib"
        if spec.char is not None:
            joblib.dump(blocks["char"], char_path)
        elif char_path.exists():
            char_path.unlink()
        out["models_replaced"] = True
    FINAL_JSON.write_text(json.dumps(out, indent=2, default=list), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("new_test", "current_test", "improvement_pp", "improved", "models_replaced")}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["search", "final"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run_search() if a.stage == "search" else run_final(a.force)
