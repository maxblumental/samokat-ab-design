from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from data import read_data
from synthetic_tests import make_synthetic_aa_test, make_synthetic_ab_test


def get_experiment_data(df: pd.DataFrame, start: datetime, days: int):
    end = start + timedelta(days=days - 1)
    return df[df['registration_date'].between(start, end)].query('ind_frod == 1').copy()


def plot_p_value_ecdf(p_values: np.ndarray, title: str):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    sns.histplot(p_values, ax=ax1, bins=20, stat='density')
    ax1.plot([0, 1], [1, 1], 'k--')
    ax1.set(xlabel='p-value', ylabel='Density')

    sns.ecdfplot(p_values, ax=ax2)
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set(xlabel='p-value', ylabel='Probability')
    ax2.grid()

    plt.savefig(title)
    plt.close()


def main():
    path = Path('dataset.csv')
    df = read_data(path)
    start = datetime(year=2023, month=7, day=1)
    recall_c = 0.43

    days = 14
    recall_t = 0.7
    exp_df = get_experiment_data(df, start, days)

    aa_p_values = make_synthetic_aa_test(exp_df, recall_c)
    ab_p_values = make_synthetic_ab_test(exp_df, recall_c, recall_t)

    p_type_1_error = np.mean(aa_p_values < 0.05)
    p_type_2_error = np.mean(ab_p_values >= 0.05)

    print(p_type_1_error)
    print(p_type_2_error)

    plot_p_value_ecdf(aa_p_values, title=f"aa_test_{days}_days_{recall_t:0.2f}_recall.png")
    plot_p_value_ecdf(ab_p_values, title=f"ab_test_{days}_days_{recall_t:0.2f}_recall.png")


if __name__ == '__main__':
    main()
