"""
Official BirdCLEF 2026 competition metric.

Macro-averaged ROC-AUC over species that have at least one positive label.
Classes with no true positives are excluded from scoring entirely.

Reference: official Kaggle scoring script (v5 utilities + score function).
"""

import numpy as np
import pandas as pd
import pandas.api.types
import sklearn.metrics

from typing import Union


class ParticipantVisibleError(Exception):
    pass


class HostVisibleError(Exception):
    pass


# ── Kaggle metric utilities (kaggle_metric_utilities.py) ─────────────────────

def treat_as_participant_error(
    error_message: str, solution: Union[pd.DataFrame, np.ndarray]
) -> bool:
    """Attempts to identify errors safe to show participants without leaking data."""
    if isinstance(solution, pd.DataFrame):
        solution_is_all_numeric = all(
            pandas.api.types.is_numeric_dtype(x) for x in solution.dtypes.values
        )
        solution_has_bools = any(
            pandas.api.types.is_bool_dtype(x) for x in solution.dtypes.values
        )
    elif isinstance(solution, np.ndarray):
        solution_is_all_numeric = pandas.api.types.is_numeric_dtype(solution)
        solution_has_bools = pandas.api.types.is_bool_dtype(solution)
    else:
        return False

    if not solution_is_all_numeric:
        return False
    for char in error_message:
        if char.isnumeric():
            return False
    if solution_has_bools:
        if "true" in error_message.lower() or "false" in error_message.lower():
            return False
    return True


def safe_call_score(metric_function, solution, submission, **metric_func_kwargs):
    """Call metric_function; re-raise as ParticipantVisibleError when safe."""
    try:
        return metric_function(solution, submission, **metric_func_kwargs)
    except Exception as err:
        error_message = str(err)
        if err.__class__.__name__ == "ParticipantVisibleError":
            raise ParticipantVisibleError(error_message)
        elif err.__class__.__name__ == "HostVisibleError":
            raise HostVisibleError(error_message)
        else:
            if treat_as_participant_error(error_message, solution):
                raise ParticipantVisibleError(error_message)
            raise


def verify_valid_probabilities(df: pd.DataFrame, df_name: str):
    """Verify that the dataframe contains valid probabilities in [0, 1]."""
    if not pandas.api.types.is_numeric_dtype(df.values):
        raise ParticipantVisibleError(
            f"All target values in {df_name} must be numeric"
        )
    if df.min().min() < 0:
        raise ParticipantVisibleError(
            f"All target values in {df_name} must be at least zero"
        )
    if df.max().max() > 1:
        raise ParticipantVisibleError(
            f"All target values in {df_name} must be no greater than one"
        )


# ── Official scoring function ─────────────────────────────────────────────────

def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
) -> float:
    """Macro-averaged ROC-AUC, ignoring classes with no true positive labels.

    Matches the official BirdCLEF 2026 Kaggle scoring script exactly.

    Args:
        solution:           Ground-truth DataFrame (row_id + species columns, binary).
        submission:         Prediction DataFrame  (row_id + species columns, scores in [0,1]).
        row_id_column_name: Column name to drop before scoring.

    Returns:
        Scalar macro ROC-AUC over species that have ≥1 positive label.
    """
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    if not pandas.api.types.is_numeric_dtype(submission.values):
        bad_dtypes = {
            x: submission[x].dtype
            for x in submission.columns
            if not pandas.api.types.is_numeric_dtype(submission[x])
        }
        raise ParticipantVisibleError(
            f"Invalid submission data types found: {bad_dtypes}"
        )

    solution_sums = solution.sum(axis=0)
    scored_columns = list(solution_sums[solution_sums > 0].index.values)
    assert len(scored_columns) > 0, "No species with positive labels found in solution."

    return safe_call_score(
        sklearn.metrics.roc_auc_score,
        solution[scored_columns].values,
        submission[scored_columns].values,
        average="macro",
    )
