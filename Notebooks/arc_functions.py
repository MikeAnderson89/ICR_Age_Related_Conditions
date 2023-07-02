import numpy as np
import pandas as pd
import scipy.stats as stats


def clean_up(df):
    df.columns = [x.strip() for x in df.columns]
    nulls = pd.DataFrame(df.isnull().sum(), columns=['nulls']).reset_index().rename(columns={'index': 'column'})
    null_columns = nulls['column'].tolist()

    for x in null_columns:
        df.loc[df[x].isnull(), x] = df[x].mean()

    return df


def data_transform(trimmed_df):
    log_cols = [
        'DI',
        'EE'
    ]

    sqrt_cols = [
        'AF'
    ]

    boxcox_cols = [
        'AB',
        'BQ',
        'DE',
        'EB',
        'FE',
        'GB'
    ]

    yeo_cols = [
        'AM',
        'GF',
        'CF'
    ]

    trimmed_df[log_cols] = np.log1p(trimmed_df[log_cols])
    trimmed_df[sqrt_cols] = np.sqrt(trimmed_df[sqrt_cols])

    for col in boxcox_cols:
        trimmed_df[col], lmbda = stats.boxcox(trimmed_df[col])

    for col in yeo_cols:
        trimmed_df[col], lmbda = stats.yeojohnson(trimmed_df[col])

    return trimmed_df


def get_used_cols(clean_df):
    used_cols = [
        'Id',
        'AF',
        'AB',
        'BQ',
        'DI',
        'FL',
        'AM',
        'CR',
        'FE',
        'DH',
        'DA',
        'BN',
        'CD',
        'BP',
        'DL',
        'EE',
        'GF',
        'DE',
        'BD',
        'CF',
        'AX',
        'FI',
        'EB',
        'GB',
        'CU',
        'EJ',
        'Class']

    trimmed_df = clean_df[used_cols]

    return trimmed_df


def prep_train(clean_df):
    df = get_used_cols(clean_df)

    for x in df.columns:
        if x not in ['Id', 'EJ', 'Class']:
            z_scores = stats.zscore(df[x])
            threshold = 3  # Adjust the threshold as per your requirement
            outliers = np.abs(z_scores) > threshold
            df = df[~outliers]

    transformed_df = data_transform(df)

    return transformed_df
