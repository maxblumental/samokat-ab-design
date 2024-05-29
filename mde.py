from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from data import read_data
from synthetic_tests import make_synthetic_aa_test, make_synthetic_ab_test


def get_experiment_data(df: pd.DataFrame, start: datetime, days: int):
    end = start + timedelta(days=days - 1)
    return df[df['registration_date'].between(start, end)].query('ind_frod == 1').copy()


def main():
    path = Path('dataset.csv')
    df = read_data(path)
    start = datetime(year=2023, month=7, day=1)
    days_values = [7, 14, 21]
    recall_t_values = np.arange(0.6, 0.81, 0.05)
    recall_c = 0.43
    alpha = 0.05

    rows = []
    for days in tqdm(days_values):
        for recall_t in recall_t_values:
            exp_df = get_experiment_data(df, start, days)

            aa_p_values = make_synthetic_aa_test(exp_df, recall_c)
            ab_p_values = make_synthetic_ab_test(exp_df, recall_c, recall_t)

            p_type_1_error = np.mean(aa_p_values < alpha)
            p_type_2_error = np.mean(ab_p_values >= alpha)

            rows.append({
                'days': days, 'recall_t': recall_t,
                'p_type_1_error': p_type_1_error, 'p_type_2_error': p_type_2_error,
            })

    pd.DataFrame(rows).to_csv('results_eq_var.csv', index=False)


if __name__ == '__main__':
    main()
    df = pd.read_csv('results_eq_var.csv')
    tmp = df.pivot_table(values=['p_type_2_error'], index=['days'], columns=['recall_t']).round(2)
    print(tmp)
    tmp.to_csv('type2_eq_var.csv', index=False)
