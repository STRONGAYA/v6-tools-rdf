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

# Make vantage6 imports optional for unit testing
try:
    from vantage6_strongaya_general.miscellaneous import PredeterminedInfoAccessor
    from vantage6_strongaya_general.general_statistics import _compute_local_missing_values
except ImportError:
    # Fallback implementations for unit testing
    class PredeterminedInfoAccessor:
        def __init__(self, df):
            self.df = df
            self._stats = {}

        def add_stat(self, name, calculation_func, **kwargs):
            self._stats[name] = calculation_func(self.df, **kwargs)

    def _compute_local_missing_values(df, placeholder=None, per_column=True, store_output_index=0, **kwargs):
        """Fallback implementation for computing missing values."""
        if placeholder is not None:
            # Check for placeholder values as well as null values
            missing_mask = df.isnull() | (df == placeholder)
        else:
            missing_mask = df.isnull()

        if per_column:
            return missing_mask.sum().to_dict()
        else:
            return missing_mask.sum().sum()


def add_missing_data_info(df: pd.DataFrame, placeholder: Any) -> None:
    """
    Add missing data information to the DataFrame using the predetermined_info attribute.

    Args:
        df (pd.DataFrame): The DataFrame to add missing data information to.
        placeholder (Any): The placeholder to use for missing data.
    """
    if not hasattr(df, "predetermined_info"):
        df.predetermined_info = PredeterminedInfoAccessor(df)
    df.predetermined_info.add_stat(
        "missing_values",
        calculation_func=_compute_local_missing_values,
        per_column=True,
        store_output_index=0,
        placeholder=placeholder,
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
    df_copy = df.copy()

    if "sub_class" in df_copy.columns:
        # Use subclass when available, otherwise use value
        df_copy[variable] = df_copy.apply(
            lambda row: (
                row["value"]
                if pd.isna(row["sub_class"]) or row["sub_class"] == ""
                else row["sub_class"]
            ),
            axis=1,
        )
        df_copy = df_copy.drop(columns=["sub_class", "value"])
    else:
        # Just rename value column to variable name
        df_copy = df_copy.rename(columns={"value": variable})

    return df_copy
