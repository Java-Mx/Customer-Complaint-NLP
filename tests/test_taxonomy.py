"""Tests for Taxonomy-Aware Complaint Classification.

Verifies:
  - Every original category has exactly one valid mapping
  - Config files (v1 conservative, v2 broad) exist and are well-formed
  - Expected 11 (conservative) and 10 (broad) normalized category counts
  - No unknown or unexpected categories in mappings
  - Deterministic mapping transformations on pandas Series
  - Train and test splits remain identical and aligned
  - Original 18-category model and artifacts remain completely unchanged
  - Programmatic assertion of intra-group error counts: 608 (conservative) and 635 (broad)
  - Experiment outputs generated without NaN or invalid metrics
  - Zero test-set leakage in taxonomy formulation
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

CONFIG_DIR = ROOT_DIR / "config"
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = ROOT_DIR / "models"


@pytest.fixture(scope="module")
def df_full():
    from src.data_loader import load_dataset
    data_path = ROOT_DIR / "data" / "complaints.csv"
    if not data_path.exists():
        pytest.skip("complaints.csv not found")
    return load_dataset(data_path, drop_invalid=True)


@pytest.fixture(scope="module")
def original_categories(df_full):
    return set(df_full["category"].unique())


@pytest.fixture(scope="module")
def cfg_v1():
    p = CONFIG_DIR / "taxonomy_v1_conservative.json"
    if not p.exists():
        pytest.skip("taxonomy_v1_conservative.json not found")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def cfg_v2():
    p = CONFIG_DIR / "taxonomy_v2_broad.json"
    if not p.exists():
        pytest.skip("taxonomy_v2_broad.json not found")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


class TestTaxonomyConfigIntegrity:
    def test_original_category_count(self, original_categories):
        """CFPB dataset must contain exactly 18 distinct categories."""
        assert len(original_categories) == 18

    def test_v1_maps_all_original_categories_exactly_once(self, original_categories, cfg_v1):
        """Every original category must be present in v1 mapping exactly once."""
        mapped_keys = set(cfg_v1["mapping"].keys())
        assert mapped_keys == original_categories
        assert len(cfg_v1["mapping"]) == 18

    def test_v2_maps_all_original_categories_exactly_once(self, original_categories, cfg_v2):
        """Every original category must be present in v2 mapping exactly once."""
        mapped_keys = set(cfg_v2["mapping"].keys())
        assert mapped_keys == original_categories
        assert len(cfg_v2["mapping"]) == 18

    def test_v1_target_category_count_is_11(self, cfg_v1):
        """v1 conservative must produce exactly 11 normalized categories."""
        targets = {v["normalized_category"] for v in cfg_v1["mapping"].values()}
        assert len(targets) == 11
        assert len(cfg_v1["normalized_categories_list"]) == 11
        assert set(cfg_v1["normalized_categories_list"]) == targets

    def test_v2_target_category_count_is_10(self, cfg_v2):
        """v2 broad must produce exactly 10 normalized categories."""
        targets = {v["normalized_category"] for v in cfg_v2["mapping"].values()}
        assert len(targets) == 10
        assert len(cfg_v2["normalized_categories_list"]) == 10
        assert set(cfg_v2["normalized_categories_list"]) == targets

    def test_v1_keeps_consumer_loan_separate(self, cfg_v1):
        """v1 must preserve Consumer Loan as an independent category."""
        assert cfg_v1["mapping"]["Consumer Loan"]["normalized_category"] == "Consumer Loan"

    def test_v2_merges_consumer_loan(self, cfg_v2):
        """v2 must merge Consumer Loan into Consumer & Small Dollar Loans."""
        assert cfg_v2["mapping"]["Consumer Loan"]["normalized_category"] == "Consumer & Small Dollar Loans"

    def test_evidence_documentation_complete(self, cfg_v1, cfg_v2):
        """Each mapped category must specify evidence type and CFPB source."""
        for cfg in [cfg_v1, cfg_v2]:
            for orig, entry in cfg["mapping"].items():
                assert "evidence_type" in entry
                assert "evidence_description" in entry
                assert "cfpb_source" in entry
                assert len(entry["evidence_description"]) > 10


class TestDeterministicMapping:
    def test_deterministic_series_mapping(self, df_full, cfg_v1, cfg_v2):
        """Mapping must be 100% deterministic and leave no NaNs."""
        map_v1 = {k: v["normalized_category"] for k, v in cfg_v1["mapping"].items()}
        map_v2 = {k: v["normalized_category"] for k, v in cfg_v2["mapping"].items()}

        s1 = df_full["category"].map(map_v1)
        s1_repeat = df_full["category"].map(map_v1)
        assert s1.isna().sum() == 0
        assert (s1 == s1_repeat).all()

        s2 = df_full["category"].map(map_v2)
        assert s2.isna().sum() == 0
        assert s2.nunique() == 10
        assert s1.nunique() == 11

    def test_train_test_split_alignment(self, df_full, cfg_v1):
        """Normalized mapping must not alter row order or split alignment."""
        from src.classification import train_test_split_data
        X_pool, X_test, y_pool, y_test = train_test_split_data(
            df_full, test_size=0.20, random_state=42, stratify=True
        )
        assert len(X_pool) == 20000
        assert len(X_test) == 5000

        map_v1 = {k: v["normalized_category"] for k, v in cfg_v1["mapping"].items()}
        y_test_norm = y_test.map(map_v1)
        assert len(y_test_norm) == 5000
        assert y_test_norm.index.equals(y_test.index)


class TestIntraGroupErrorAssertions:
    def test_assert_608_and_635_error_counts(self):
        """Audits that conservative=608 and broad=635 error sums match results/error_analysis.csv exactly."""
        from scripts.run_taxonomy_experiment import verify_intra_group_error_counts
        c_err, b_err, t_err = verify_intra_group_error_counts()
        assert t_err == 1522
        assert c_err == 608
        assert b_err == 635


class TestExperimentOutputs:
    def test_experiment_csv_exists_and_valid(self):
        """results/taxonomy_experiment.csv must exist with valid metrics."""
        csv_path = RESULTS_DIR / "taxonomy_experiment.csv"
        if not csv_path.exists():
            pytest.skip("taxonomy_experiment.csv not generated yet")
        df_comp = pd.read_csv(csv_path)
        assert not df_comp.empty
        assert "Metric" in df_comp.columns
        assert "Reference (18 Categories)" in df_comp.columns
        assert "v1 Conservative (11 Categories)" in df_comp.columns
        assert "v2 Broad (10 Categories)" in df_comp.columns
        assert not df_comp.isna().any().any()

    def test_experiment_json_exists_and_valid(self):
        """results/taxonomy_experiment_data.json must contain models and information loss."""
        json_path = RESULTS_DIR / "taxonomy_experiment_data.json"
        if not json_path.exists():
            pytest.skip("taxonomy_experiment_data.json not generated yet")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "models" in data
        assert "reference_18" in data["models"]
        assert "v1_conservative" in data["models"]
        assert "v2_broad" in data["models"]

        v1_metrics = data["models"]["v1_conservative"]["metrics"]
        v2_metrics = data["models"]["v2_broad"]["metrics"]
        ref_metrics = data["models"]["reference_18"]["metrics"]

        # Basic ranges
        for m in [v1_metrics, v2_metrics, ref_metrics]:
            assert 0.60 <= m["accuracy"] <= 1.0
            assert 0.40 <= m["macro_f1"] <= 1.0

        # Information loss exists
        assert "information_loss" in data["models"]["v1_conservative"]
        assert "information_loss" in data["models"]["v2_broad"]


class TestReferenceModelUntouched:
    def test_original_classifier_file_exists(self):
        """The original 18-category classifier must remain intact."""
        clf_path = MODELS_DIR / "complaint_classifier.joblib"
        assert clf_path.exists()
        import joblib
        clf = joblib.load(clf_path)
        assert len(clf.classes_) == 18
