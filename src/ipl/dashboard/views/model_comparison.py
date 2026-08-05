"""Model comparison page: how every algorithm scored, and why one was chosen."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ...models.persistence import ARTIFACTS, load_metrics
from ...models.registry import describe_availability, model_family
from ..theme import CATEGORICAL, bar_chart, line_chart, series_palette
from ._common import metric_row, page_header, show_table

TASK_LABELS = {
    "winner": "Match winner",
    "score": "First-innings score",
    "chase": "Chase success",
    "pom": "Player of the Match",
}

# Which metric decides each task, and whether bigger is better.
PRIMARY = {
    "winner": ("roc_auc", "ROC-AUC", True),
    "score": ("rmse", "RMSE (runs)", False),
    "chase": ("roc_auc", "ROC-AUC", True),
    "pom": ("roc_auc", "ROC-AUC", True),
}


def render() -> None:
    """Render the model comparison page."""
    page_header(
        "Model Comparison",
        "Every algorithm, scored on held-out seasons the models never saw during training.",
    )

    available = describe_availability()
    missing = [name for name, ok in available.items() if not ok]
    if missing:
        st.warning(
            f"These optional libraries are not installed, so they were excluded "
            f"from the comparison: {', '.join(missing)}."
        )

    metrics = {task: load_metrics(name) for task, name in ARTIFACTS.items()}
    if not any(metrics.values()):
        st.info(
            "No trained models found. Run `python scripts/train_models.py` "
            "or use the **Admin** page."
        )
        return

    st.markdown(
        """
        **How these numbers were produced.** Models are trained on early seasons
        and scored on the most recent ones, which they never see during fitting.
        A random train/test split would leak the future into the past and inflate
        every figure here.
        """
    )

    with st.expander(
        "Why the pre-match winner model scores near 0.50 — and why that is the "
        "honest answer",
        expanded=False,
    ):
        st.markdown(
            """
            The match-winner model lands at a ROC-AUC of roughly **0.50-0.55** on
            held-out seasons. That is close to a coin toss, and it is a *finding*
            rather than a defect.

            **The evidence.** The model was re-evaluated at six independent
            cut-off points, holding out everything from the last two seasons
            (144 matches) up to the last eight (541 matches). Every window gave
            an AUC between 0.50 and 0.55. A result that stable across 541 test
            matches is not sampling noise.

            **Why.** No pre-match feature in the dataset correlates with the
            result at more than |r| ≈ 0.09 — not recent form, not head-to-head,
            not home advantage, not the Playing XI's career batting average. The
            home side wins about 52% of IPL matches, so there is very little to
            predict from before the toss. T20 is a short, high-variance format
            where a single over swings the result.

            **What was done about it.** The features include Playing XI strength
            built from each selected player's career record, era-aware scoring
            levels, rest days, venue records and rolling form; the training set
            is mirrored so each fixture is seen from both sides, removing any
            advantage from the arbitrary "team 1 is listed first" convention.
            These changes lifted the model from *below* chance to chance. They
            did not manufacture signal that is not in the data.

            **Where the models do work.** The chase model reaches **AUC ≈ 0.90**
            and the Player-of-the-Match ranker picks the actual winner in about
            **56%** of matches from a field of ~22. Both see what actually
            happened in the match. The contrast between them and the pre-match
            model is the useful lesson here.

            A model that reported 75% accuracy on this task would be leaking the
            future into its training data. The number below is the real one.
            """
        )
    st.write("")

    tabs = st.tabs([TASK_LABELS[t] for t in ARTIFACTS if metrics.get(t)])
    tasks = [t for t in ARTIFACTS if metrics.get(t)]

    for tab, task in zip(tabs, tasks):
        with tab:
            _task_panel(task, metrics[task])


def _task_panel(task: str, payload: dict) -> None:
    """Render one task's comparison table, chart and diagnostics."""
    rows = payload.get("metrics") or []
    if not rows:
        st.caption("No metrics recorded for this task.")
        return

    frame = pd.DataFrame(rows)
    metric_key, metric_label, higher_is_better = PRIMARY.get(task, ("roc_auc", "ROC-AUC", True))
    best = payload.get("best_model")

    header = [
        ("Best model", best or "—", None),
        ("Training rows", f"{payload.get('train_rows', 0):,}", None),
        ("Test rows", f"{payload.get('test_rows', 0):,}", None),
    ]
    if payload.get("test_seasons"):
        seasons = payload["test_seasons"]
        header.append(("Held-out seasons", f"{min(seasons)}–{max(seasons)}", None))
    elif payload.get("top1_accuracy") is not None:
        header.append(("Top-1 accuracy", f"{payload['top1_accuracy']:.1%}", None))
    metric_row(header)
    st.write("")

    # --- comparison chart on the deciding metric ---
    chart_frame = frame.dropna(subset=[metric_key]).copy()
    if not chart_frame.empty:
        chart_frame = chart_frame.sort_values(metric_key, ascending=higher_is_better)
        # The winning model is highlighted; the rest share a recessive slot.
        colors = [
            CATEGORICAL[0] if name == best else "#c3c2b7"
            for name in chart_frame["model"]
        ]
        st.plotly_chart(
            bar_chart(
                chart_frame, "model", metric_key,
                title=f"{metric_label} by model — higher is better"
                if higher_is_better else f"{metric_label} by model — lower is better",
                colors=colors, orientation="h", height=360,
                text_format=".4f", x_title=metric_label, y_title="",
            ),
            use_container_width=True,
        )
        st.caption(f"The selected model ({best}) is highlighted.")

    # --- full metric table ---
    display = frame.copy()
    display["family"] = display["model"].map(model_family)
    display["selected"] = display["model"] == best

    if task == "score":
        columns = {
            "model": "Model", "family": "Family", "rmse": "RMSE", "mae": "MAE",
            "r2": "R²", "within_10": "Within 10 runs", "within_20": "Within 20 runs",
            "train_seconds": "Fit (s)", "selected": "Selected",
        }
    else:
        columns = {
            "model": "Model", "family": "Family", "accuracy": "Accuracy",
            "precision": "Precision", "recall": "Recall", "f1": "F1",
            "roc_auc": "ROC-AUC", "log_loss": "Log loss", "brier": "Brier",
            "train_seconds": "Fit (s)", "selected": "Selected",
        }

    available = {k: v for k, v in columns.items() if k in display.columns}
    show_table(display[list(available)].rename(columns=available))

    if task != "score":
        st.caption(
            "Accuracy, precision, recall and F1 use a 0.5 threshold. ROC-AUC is "
            "threshold-independent and is what selects the model. Log loss and "
            "Brier score measure whether the probabilities themselves are honest."
        )
    else:
        st.caption(
            "'Within 10 runs' is the share of held-out innings the model called "
            "to within ten runs — a more intuitive read than RMSE alone."
        )

    # --- calibration ---
    calibration = payload.get("calibration")
    if calibration:
        st.subheader("Is the model honest about its confidence?")
        cal = pd.DataFrame(calibration)
        if not cal.empty:
            cal["predicted_pct"] = (cal["predicted"] * 100).round(1)
            cal["observed_pct"] = (cal["observed"] * 100).round(1)
            st.plotly_chart(
                line_chart(
                    cal, "bucket",
                    {"Predicted": "predicted_pct", "Observed": "observed_pct"},
                    title="Predicted vs observed win rate",
                    x_title="Confidence bucket", y_title="%",
                ),
                use_container_width=True,
            )
            st.caption(
                "The two lines should track each other. Where 'observed' sits below "
                "'predicted', the model is overconfident in that band."
            )
            show_table(
                cal[["bucket", "count", "predicted_pct", "observed_pct"]].rename(
                    columns={
                        "bucket": "Confidence", "count": "Predictions",
                        "predicted_pct": "Predicted %", "observed_pct": "Observed %",
                    }
                )
            )

    # --- feature importance ---
    importance = payload.get("importance")
    if importance:
        st.subheader("What the model leans on")
        frame_imp = pd.DataFrame(importance).head(15)
        if not frame_imp.empty:
            frame_imp["importance_pct"] = (frame_imp["importance"] * 100).round(2)
            st.plotly_chart(
                bar_chart(
                    frame_imp.sort_values("importance_pct"), "feature", "importance_pct",
                    title="Top features by relative importance",
                    orientation="h", height=max(360, 26 * len(frame_imp)),
                    colors=[CATEGORICAL[2]] * len(frame_imp),
                    text_format=".2f", x_title="Relative importance (%)", y_title="",
                ),
                use_container_width=True,
            )
            show_table(
                frame_imp[["feature", "importance_pct"]].rename(
                    columns={"feature": "Feature", "importance_pct": "Importance (%)"}
                )
            )
            st.caption(
                "Importance is the model's own attribution, normalised to sum to 100%. "
                "It shows what the model uses, not what causes a result."
            )

    if payload.get("trained_at"):
        st.caption(f"Trained at {payload['trained_at']} UTC.")
