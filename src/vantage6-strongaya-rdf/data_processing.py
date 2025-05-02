"""
------------------------------------------------------------------------------
RDF/SPARQL Data Processing Functions

File organisation:
- Missing data information determination and storage (add_missing_data_info)
- Subclass information extraction (extract_subclass_info)
------------------------------------------------------------------------------
"""

import pandas as pd

from typing import Any

from vantage6_strongaya_general.miscellaneous import PredeterminedInfoAccessor
from vantage6_strongaya_general.general_statistics import _compute_local_missing_values


def add_missing_data_info(df: pd.DataFrame, placeholder: Any) -> None:
    """
    Add missing data information to the DataFrame using the predetermined_info attribute.

    Args:
        df (pd.DataFrame): The DataFrame to add missing data information to.
        placeholder (Any): The placeholder to use for missing data.
    """
    if not hasattr(df, 'predetermined_info'):
        df.predetermined_info = PredeterminedInfoAccessor(df)
    df.predetermined_info.add_stat(
        'missing_values',
        calculation_func=_compute_local_missing_values,
        per_column=True,
        store_output_index=0,
        placeholder=placeholder
    )


def extract_subclass_info(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    """
    Extract subclass information from the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        variable (str): The variable name to extract subclass information for.

    Returns:
        pd.DataFrame: The DataFrame with subclass information where it is available.
    """
    if 'sub_class' in df.columns:
        df[variable] = df.apply(
            lambda row: row['value'] if pd.isna(row['sub_class']) or row['sub_class'] == '' else row['sub_class'],
            axis=1
        )
        df = df.drop(columns=['sub_class', 'value'])
    else:
        df = df.rename(columns={'value': variable})
    return df
