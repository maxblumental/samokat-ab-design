from pathlib import Path

import pandas as pd


def read_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=';')
    df['registration_date'] = pd.to_datetime(df['registration_date'], format="%d.%m.%Y")
    df['activation_date'] = pd.to_datetime(df['activation_date'], format="%d.%m.%Y")

    df = df.query('registration_date > "1970-01-01"')
    df = df.query('activation_date.isna() or (registration_date <= activation_date)')
    df = df.dropna(subset=['ind_frod'])

    return df
