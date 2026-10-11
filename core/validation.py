"""Validation and data loading logic for AuditLens."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from core.config import DEFAULT_CONFIG, AuditLensConfig
from core.errors import InvalidInputError
from core.log_setup import logger


@dataclass
class ValidationWarning:
    """Warning for non-fatal dataset quality issues."""

    message: str
    column: Optional[str] = None
    group_value: Optional[str] = None


@dataclass
class AuditDataset:
    """Validated, typed container for credit model audit data."""

    data: pd.DataFrame
    score: pd.Series
    outcome: pd.Series
    group_columns: List[str] = field(default_factory=list)
    group_tags: Dict[str, str] = field(default_factory=dict)
    timestamp_column: Optional[str] = None
    warnings: List[ValidationWarning] = field(default_factory=list)
    unreliable_groups: Set[Tuple[str, Any]] = field(default_factory=set)

    @property
    def n_rows(self) -> int:
        """Total row count."""
        return len(self.data)


def _check_required_columns(df: pd.DataFrame) -> None:
    """Verifies that mandatory columns exist in DataFrame."""
    required = {"score", "outcome"}
    missing = required - set(df.columns)
    if missing:
        msg = f"Missing required column(s): {sorted(list(missing))}."
        hint = "CSV must include 'score' (probability) and 'outcome' (binary 0/1)."
        logger.error(msg)
        raise InvalidInputError(msg, fix_hint=hint)


def _check_no_nulls(df: pd.DataFrame) -> None:
    """Ensures no null values exist in required columns."""
    for col in ("score", "outcome"):
        null_count = df[col].isnull().sum()
        if null_count > 0:
            msg = f"Column '{col}' contains {null_count} missing/null values."
            hint = f"Filter or impute null entries in '{col}' before uploading."
            logger.error(msg)
            raise InvalidInputError(msg, fix_hint=hint)


def _check_row_count(df: pd.DataFrame, min_rows: int) -> None:
    """Verifies dataset has at least the minimum required rows."""
    if len(df) < min_rows:
        msg = f"Dataset contains {len(df)} rows, minimum required is {min_rows}."
        hint = f"Provide an audit sample with at least {min_rows} observations."
        logger.error(msg)
        raise InvalidInputError(msg, fix_hint=hint)


def _check_values(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Validates score is in [0, 1] and outcome is in {0, 1}."""
    try:
        score = pd.to_numeric(df["score"], errors="raise").astype(float)
        outcome = pd.to_numeric(df["outcome"], errors="raise").astype(int)
    except Exception as exc:
        msg = f"Non-numeric values found in score or outcome: {exc}."
        hint = "Ensure 'score' is numeric floats and 'outcome' is numeric 0 or 1."
        logger.error(msg)
        raise InvalidInputError(msg, fix_hint=hint) from exc

    if (score < 0.0).any() or (score > 1.0).any():
        msg = "Column 'score' contains values outside [0.0, 1.0]."
        hint = "Normalize model scores to predicted probabilities in [0.0, 1.0]."
        logger.error(msg)
        raise InvalidInputError(msg, fix_hint=hint)

    unique_outcomes = set(outcome.unique())
    if not unique_outcomes.issubset({0, 1}):
        msg = f"Column 'outcome' contains invalid classes: {unique_outcomes}."
        hint = "Outcome column must only contain 0 (repaid) or 1 (defaulted)."
        logger.error(msg)
        raise InvalidInputError(msg, fix_hint=hint)

    if len(unique_outcomes) < 2:
        msg = f"Outcome must contain both classes (0 and 1), found: {unique_outcomes}."
        hint = "Provide audit data containing both positive and negative outcomes."
        logger.error(msg)
        raise InvalidInputError(msg, fix_hint=hint)

    return score, outcome


def _evaluate_group_warnings(
    df: pd.DataFrame,
    group_columns: List[str],
    threshold: int,
) -> Tuple[List[ValidationWarning], Set[Tuple[str, Any]]]:
    """Checks for small sample groups below threshold without failing."""
    warnings: List[ValidationWarning] = []
    unreliable: Set[Tuple[str, Any]] = set()

    for col in group_columns:
        if col not in df.columns:
            continue
        counts = df[col].value_counts(dropna=False)
        for val, count in counts.items():
            if count < threshold:
                val_str = str(val)
                msg = (
                    f"Group '{col}={val_str}' has {count} rows "
                    f"(< {threshold}). Unreliable (small sample)."
                )
                logger.warning(msg)
                warnings.append(
                    ValidationWarning(message=msg, column=col, group_value=val_str)
                )
                unreliable.add((col, val))

    return warnings, unreliable


def validate_and_load(
    df: pd.DataFrame,
    group_columns: Optional[List[str]] = None,
    group_tags: Optional[Dict[str, str]] = None,
    timestamp_column: Optional[str] = None,
    config: Optional[AuditLensConfig] = None,
) -> AuditDataset:
    """Validates raw DataFrame according to audit contracts and returns AuditDataset.

    Args:
        df: Raw input DataFrame.
        group_columns: List of columns identified as demographic/operational groups.
        group_tags: Mapping of group column name to 'operational' or 'audit-only'.
        timestamp_column: Optional timestamp column name for drift monitoring.
        config: Audit configuration. Defaults to DEFAULT_CONFIG.

    Returns:
        AuditDataset instance.

    Raises:
        InvalidInputError: If required columns, bounds, row count, or types fail.
    """
    cfg = config or DEFAULT_CONFIG
    logger.info(
        f"Validating dataset with {len(df)} rows and {len(df.columns)} columns."
    )

    if not isinstance(df, pd.DataFrame):
        raise InvalidInputError("Input is not a valid pandas DataFrame.")

    _check_required_columns(df)
    _check_no_nulls(df)
    _check_row_count(df, cfg.min_rows)
    score, outcome = _check_values(df)

    clean_df = df.copy()
    clean_df["score"] = score
    clean_df["outcome"] = outcome

    valid_groups = [c for c in (group_columns or []) if c in clean_df.columns]
    for c in valid_groups:
        clean_df[c] = clean_df[c].astype(str)
    tags = {c: (group_tags or {}).get(c, "audit-only") for c in valid_groups}

    if timestamp_column and timestamp_column in clean_df.columns:
        try:
            clean_df[timestamp_column] = pd.to_datetime(clean_df[timestamp_column])
        except Exception as exc:
            logger.warning(
                f"Could not parse timestamp column {timestamp_column}: {exc}"
            )
            timestamp_column = None
    else:
        timestamp_column = None

    warnings, unreliable = _evaluate_group_warnings(
        clean_df, valid_groups, cfg.small_group_threshold
    )

    logger.info(
        f"Validation passed: {len(clean_df)} rows, {len(valid_groups)} groups, "
        f"{len(warnings)} warnings."
    )

    return AuditDataset(
        data=clean_df,
        score=clean_df["score"],
        outcome=clean_df["outcome"],
        group_columns=valid_groups,
        group_tags=tags,
        timestamp_column=timestamp_column,
        warnings=warnings,
        unreliable_groups=unreliable,
    )
