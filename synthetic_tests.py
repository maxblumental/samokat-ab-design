import numpy as np
import pandas as pd
from scipy import stats


def make_synthetic_aa_test(df: pd.DataFrame, recall_c: float) -> np.ndarray:
    arr = []
    ind_frod = df['ind_frod'].values
    for _ in range(10_000):
        split = np.random.binomial(n=1, p=0.5, size=len(df))
        control_frauds = int(ind_frod @ (1 - split))
        test_frauds = int(ind_frod @ split)
        control = np.where(np.random.rand(control_frauds) < recall_c, 1, 0)
        test = np.where(np.random.rand(test_frauds) < recall_c, 1, 0)
        arr.append(stats.ttest_ind(control, test).pvalue)
    return np.array(arr)


def make_synthetic_ab_test(df: pd.DataFrame, recall_c: float, recall_t: float) -> np.ndarray:
    arr = []
    ind_frod = df['ind_frod'].values
    for _ in range(10_000):
        split = np.random.binomial(n=1, p=0.5, size=len(df))
        control_frauds = int(ind_frod @ (1 - split))
        test_frauds = int(ind_frod @ split)
        control = np.where(np.random.rand(control_frauds) < recall_c, 1, 0)
        test = np.where(np.random.rand(test_frauds) < recall_t, 1, 0)
        arr.append(stats.ttest_ind(control, test).pvalue)
    return np.array(arr)
