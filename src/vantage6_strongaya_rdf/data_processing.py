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

# Optional import for strongaya general - only needed for full vantage6 integration
try:
    from vantage6_strongaya_general.miscellaneous import PredeterminedInfoAccessor
    from vantage6_strongaya_general.general_statistics import _compute_local_missing_values
    STRONGAYA_GENERAL_AVAILABLE = True
except ImportError:
    STRONGAYA_GENERAL_AVAILABLE = False
    # Provide fallback implementations
    class PredeterminedInfoAccessor:
        def __init__(self, df):
            self._df = df
            self._stats = {}
            
        def add_stat(self, name, calculation_func=None, per_column=False, 
                     store_output_index=None, **kwargs):
            """Add statistical information (fallback implementation)."""
            if calculation_func:
                try:
                    result = calculation_func(self._df, **kwargs)
                    self._stats[name] = result
                except Exception:
                    # Fallback calculation
                    self._stats[name] = {"computed": True}
    
    def _compute_local_missing_values(df, placeholder=None):
        """Simple fallback: count None/NaN values."""
        missing_info = {}
        for column in df.columns:
            if column != 'patient_id':
                missing_count = df[column].isna().sum()
                if placeholder is not None:
                    missing_count += (df[column] == placeholder).sum()
                missing_info[column] = {
                    'missing_count': missing_count,
                    'total_count': len(df)
                }
        return missing_info


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
    if "sub_class" in df.columns:
        df[variable] = df.apply(
            lambda row: (
                row["value"]
                if pd.isna(row["sub_class"]) or row["sub_class"] == ""
                else row["sub_class"]
            ),
            axis=1,
        )
        df = df.drop(columns=["sub_class", "value"])
    else:
        df = df.rename(columns={"value": variable})
    return df
