"""Dataset splitting logic for AuditLens."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from core.config import DEFAULT_CONFIG, AuditLensConfig
from core.errors import InvalidInputError
from core.log_setup import logger
from core.validation import AuditDataset


@dataclass
class DatasetSplits:
    """Container holding stratified calibration and test sets."""

    cal_data: pd.DataFrame
    test_data: pd.DataFrame
    cal_score: pd.Series
    cal_outcome: pd.Series
    test_score: pd.Series
    test_outcome: pd.Series
    cal_indices: np.ndarray
    test_indices: np.ndarray

    @property
    def n_cal(self) -> int:
        """Calibration sample size."""
        return len(self.cal_data)

    @property
    def n_test(self) -> int:
        """Test sample size."""
        return len(self.test_data)


def create_stratified_split(
    dataset: AuditDataset,
    config: Optional[AuditLensConfig] = None,
    cal_ratio: Optional[float] = None,
    seed: Optional[int] = None,
) -> DatasetSplits:
    """Splits dataset into stratified calibration and test sets (50/50 by default).

    Args:
        dataset: Validated AuditDataset.
        config: Audit configuration. Defaults to DEFAULT_CONFIG.
        cal_ratio: Fraction of data for calibration. Defaults to config.calibration_split_ratio.
        seed: Random seed. Defaults to config.random_seed.

    Returns:
        DatasetSplits with disjoint calibration and test subsets.

    Raises:
        InvalidInputError: If dataset is too small or cannot be stratified.
    """
    cfg = config or DEFAULT_CONFIG
    ratio = cal_ratio if cal_ratio is not None else cfg.calibration_split_ratio
    rng_seed = seed if seed is not None else cfg.random_seed
    test_ratio = 1.0 - ratio

    logger.info(
        f"Splitting dataset of size {dataset.n_rows} with calibration ratio "
        f"{ratio:.2f} and seed {rng_seed}."
    )

    if dataset.n_rows < 2:
        raise InvalidInputError(
            "Dataset must contain at least 2 rows to split.",
            fix_hint="Upload a dataset meeting minimum row requirements.",
        )

    try:
        cal_idx, test_idx = train_test_split(
            np.arange(dataset.n_rows),
            test_size=test_ratio,
            stratify=dataset.outcome.to_numpy(),
            random_state=rng_seed,
            shuffle=True,
        )
    except Exception as exc:
        msg = f"Failed to perform stratified split: {exc}"
        logger.error(msg)
        raise InvalidInputError(
            msg, fix_hint="Check outcome class distribution for extreme imbalances."
        ) from exc

    cal_df = dataset.data.iloc[cal_idx].copy().reset_index(drop=True)
    test_df = dataset.data.iloc[test_idx].copy().reset_index(drop=True)

    cal_score = cal_df["score"]
    cal_outcome = cal_df["outcome"]
    test_score = test_df["score"]
    test_outcome = test_df["outcome"]

    logger.info(
        f"Split complete: cal_n={len(cal_df)} (default_rate={cal_outcome.mean():.4f}), "
        f"test_n={len(test_df)} (default_rate={test_outcome.mean():.4f})."
    )

    return DatasetSplits(
        cal_data=cal_df,
        test_data=test_df,
        cal_score=cal_score,
        cal_outcome=cal_outcome,
        test_score=test_score,
        test_outcome=test_outcome,
        cal_indices=cal_idx,
        test_indices=test_idx,
    )
