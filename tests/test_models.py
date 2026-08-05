"""Tests for the model layer: registry, evaluation, selection and playoffs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split

from ipl.models.evaluate import (
    ClassificationMetrics,
    RegressionMetrics,
    calibration_table,
    evaluate_classifier,
    evaluate_regressor,
    feature_importance,
    metrics_to_frame,
    pick_best_model,
)
from ipl.models.playoffs import simulate_playoff_qualification
from ipl.models.registry import (
    build_classifiers,
    build_regressors,
    describe_availability,
    model_family,
    needs_scaling,
)


@pytest.fixture()
def binary_data():
    X, y = make_classification(
        n_samples=400, n_features=8, n_informative=5, random_state=0
    )
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    return train_test_split(frame, y, test_size=0.3, random_state=0)


@pytest.fixture()
def regression_data():
    X, y = make_regression(n_samples=400, n_features=8, noise=12.0, random_state=0)
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    return train_test_split(frame, y, test_size=0.3, random_state=0)


class TestRegistry:
    def test_the_three_always_available_classifiers_are_present(self):
        models = build_classifiers()
        for name in ("Logistic Regression", "Random Forest", "Gradient Boosting"):
            assert name in models

    def test_regressors_include_a_linear_baseline(self):
        assert "Ridge Regression" in build_regressors()

    def test_only_linear_models_need_scaling(self):
        assert needs_scaling("Logistic Regression") is True
        assert needs_scaling("Ridge Regression") is True
        assert needs_scaling("Random Forest") is False
        assert needs_scaling("XGBoost") is False

    def test_model_family_labels(self):
        assert model_family("Logistic Regression") == "Linear"
        assert model_family("Random Forest") == "Bagging"
        assert model_family("CatBoost") == "Boosting"

    def test_availability_report(self):
        availability = describe_availability()
        assert set(availability) == {"xgboost", "lightgbm", "catboost"}
        assert all(isinstance(v, bool) for v in availability.values())

    def test_random_state_is_applied(self):
        models = build_classifiers(random_state=7)
        assert models["Random Forest"].random_state == 7


class TestClassificationMetrics:
    def test_all_metrics_are_produced(self, binary_data):
        X_train, X_test, y_train, y_test = binary_data
        model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
        metrics = evaluate_classifier(model, X_test, y_test, name="LogReg")

        assert isinstance(metrics, ClassificationMetrics)
        for value in (metrics.accuracy, metrics.precision, metrics.recall,
                      metrics.f1, metrics.roc_auc):
            assert 0.0 <= value <= 1.0
        assert metrics.support == len(y_test)
        assert len(metrics.confusion) == 2

    def test_a_model_with_signal_beats_chance(self, binary_data):
        X_train, X_test, y_train, y_test = binary_data
        model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
        metrics = evaluate_classifier(model, X_test, y_test, name="LogReg")
        assert metrics.roc_auc > 0.7

    def test_single_class_test_set_yields_nan_auc(self):
        X = pd.DataFrame(np.random.rand(20, 3), columns=list("abc"))
        y = np.zeros(20, dtype=int)
        model = LogisticRegression().fit(
            pd.DataFrame(np.random.rand(20, 3), columns=list("abc")),
            np.array([0] * 10 + [1] * 10),
        )
        metrics = evaluate_classifier(model, X, y, name="LogReg")
        assert np.isnan(metrics.roc_auc)

    def test_summary_is_readable(self, binary_data):
        X_train, X_test, y_train, y_test = binary_data
        model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
        assert "auc=" in evaluate_classifier(model, X_test, y_test, name="LogReg").summary()


class TestRegressionMetrics:
    def test_all_metrics_are_produced(self, regression_data):
        X_train, X_test, y_train, y_test = regression_data
        model = Ridge().fit(X_train, y_train)
        metrics = evaluate_regressor(model, X_test, y_test, name="Ridge")

        assert isinstance(metrics, RegressionMetrics)
        assert metrics.rmse > 0
        assert metrics.mae > 0
        assert metrics.r2 > 0.5  # a linear fit on linear data should be strong
        assert 0.0 <= metrics.within_10 <= 1.0

    def test_rmse_is_at_least_mae(self, regression_data):
        X_train, X_test, y_train, y_test = regression_data
        model = Ridge().fit(X_train, y_train)
        metrics = evaluate_regressor(model, X_test, y_test, name="Ridge")
        assert metrics.rmse >= metrics.mae


class TestModelSelection:
    def _classification(self) -> list[ClassificationMetrics]:
        return [
            ClassificationMetrics("A", 0.7, 0.7, 0.7, 0.7, 0.80, 0.5, 0.2, 100, 0.5),
            ClassificationMetrics("B", 0.6, 0.6, 0.6, 0.6, 0.91, 0.5, 0.2, 100, 0.5),
            ClassificationMetrics("C", 0.8, 0.8, 0.8, 0.8, 0.75, 0.5, 0.2, 100, 0.5),
        ]

    def test_classification_picks_highest_auc(self):
        assert pick_best_model(self._classification()) == "B"

    def test_can_select_on_another_metric(self):
        assert pick_best_model(self._classification(), metric="accuracy") == "C"

    def test_regression_picks_lowest_rmse(self):
        metrics = [
            RegressionMetrics("A", 20.0, 15.0, 0.5, 0.0, 0.4, 0.7, 100),
            RegressionMetrics("B", 15.0, 12.0, 0.6, 0.0, 0.5, 0.8, 100),
        ]
        assert pick_best_model(metrics) == "B"

    def test_nan_scores_are_ignored(self):
        metrics = self._classification()
        metrics[1].roc_auc = float("nan")
        assert pick_best_model(metrics) == "A"

    def test_ties_resolve_deterministically(self):
        metrics = [
            ClassificationMetrics("Zebra", 0.7, 0.7, 0.7, 0.7, 0.8, 0.5, 0.2, 100, 0.5),
            ClassificationMetrics("Alpha", 0.7, 0.7, 0.7, 0.7, 0.8, 0.5, 0.2, 100, 0.5),
        ]
        assert pick_best_model(metrics) == "Alpha"
        assert pick_best_model(list(reversed(metrics))) == "Alpha"

    def test_empty_returns_none(self):
        assert pick_best_model([]) is None


class TestDiagnostics:
    def test_metrics_frame_drops_the_confusion_matrix(self, binary_data):
        X_train, X_test, y_train, y_test = binary_data
        model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
        frame = metrics_to_frame([evaluate_classifier(model, X_test, y_test, name="M")])
        assert "confusion" not in frame.columns
        assert "roc_auc" in frame.columns

    def test_calibration_buckets_sum_to_the_sample(self, binary_data):
        X_train, X_test, y_train, y_test = binary_data
        model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
        table = calibration_table(model, X_test, y_test)
        assert table["count"].sum() == len(y_test)
        assert (table["predicted"].between(0, 1)).all()

    def test_feature_importance_normalises_to_one(self, binary_data):
        from sklearn.ensemble import RandomForestClassifier

        X_train, _X_test, y_train, _y_test = binary_data
        model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X_train, y_train)
        importance = feature_importance(model, list(X_train.columns))
        assert not importance.empty
        assert importance["importance"].sum() == pytest.approx(1.0, abs=1e-6)

    def test_feature_importance_handles_a_mismatch(self):
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=5).fit(
            np.random.rand(30, 4), np.random.randint(0, 2, 30)
        )
        assert feature_importance(model, ["only", "two"]).empty


class TestPlayoffSimulation:
    def test_probabilities_are_percentages(self, sample_matches, sample_innings):
        projection = simulate_playoff_qualification(
            sample_matches, sample_innings, 2023, simulations=200
        )
        assert not projection.table.empty
        assert projection.table["qualification_pct"].between(0, 100).all()

    def test_completed_season_gives_a_decided_table(self, sample_matches, sample_innings):
        # Every fixture in the sample is complete, so nothing is left to simulate.
        projection = simulate_playoff_qualification(
            sample_matches, sample_innings, 2023, simulations=100
        )
        assert projection.matches_remaining == 0
        assert set(projection.table["qualification_pct"].unique()) <= {0.0, 100.0}

    def test_unknown_season_returns_empty(self, sample_matches, sample_innings):
        projection = simulate_playoff_qualification(
            sample_matches, sample_innings, 1999, simulations=50
        )
        assert projection.table.empty

    def test_simulation_is_reproducible(self, sample_matches, sample_innings):
        scheduled = sample_matches.copy()
        scheduled.loc[scheduled.index[-1], "is_completed"] = False
        scheduled.loc[scheduled.index[-1], "winner"] = None

        first = simulate_playoff_qualification(
            scheduled, sample_innings, 2024, simulations=500, random_state=7
        )
        second = simulate_playoff_qualification(
            scheduled, sample_innings, 2024, simulations=500, random_state=7
        )
        pd.testing.assert_frame_equal(first.table, second.table)
