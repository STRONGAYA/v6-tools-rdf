"""
------------------------------------------------------------------------------
RDF/SPARQL Data Processing Functions

File organisation:
- Missing data information determination and storage (add_missing_data_info)
- Subclass information extraction (extract_subclass_info)
- NULL value cleaning (clean_null_values)
------------------------------------------------------------------------------
"""

import pandas as pd
import re
from typing import Any
from vantage6_strongaya_general.miscellaneous import PredeterminedInfoAccessor
from vantage6_strongaya_general.general_statistics import _compute_local_missing_values


def clean_null_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean NULL values in the DataFrame by converting various NULL representations to pd.NA.
    
    This function handles cases where NULL values are represented as:
    - String "NULL"
    - String "['NULL']" (string representation of list containing NULL)
    - Empty strings
    - Other common NULL representations
    
    Args:
        df (pd.DataFrame): The DataFrame to clean NULL values from.
        
    Returns:
        pd.DataFrame: The DataFrame with cleaned NULL values.
    """
    # Create a copy to avoid modifying the original DataFrame
    df_cleaned = df.copy()
    
    # Define patterns for NULL value detection
    null_patterns = [
        r'^NULL$',  # Simple NULL string
        r"^\['NULL'\]$",  # String representation of list with NULL
        r'^null$',  # Lowercase null
        r"^\['null'\]$",  # String representation of list with lowercase null
        r'^\s*$',  # Empty or whitespace-only strings
        r'^NaN$',  # NaN string
        r'^nan$',  # lowercase nan
        r'^None$',  # None string
        r'^none$',  # lowercase none
    ]
    
    # Combine patterns into a single regex
    combined_pattern = re.compile('|'.join(null_patterns), re.IGNORECASE)
    
    # Apply cleaning to all columns except patient_id
    for col in df_cleaned.columns:
        if col != "patient_id":
            # Convert string NULL representations to pd.NA
            df_cleaned[col] = df_cleaned[col].apply(
                lambda x: pd.NA if isinstance(x, str) and combined_pattern.match(x) else x
            )
            
            # Also handle the case where x might be a list containing NULL strings
            df_cleaned[col] = df_cleaned[col].apply(
                lambda x: pd.NA if isinstance(x, list) and len(x) == 1 and combined_pattern.match(str(x[0])) else x
            )
    
    return df_cleaned


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
                row["any_value"]
                if pd.isna(row["sub_class"]) or row["sub_class"] == ""
                else row["sub_class"]
            ),
            axis=1,
        )
        df = df.drop(columns=["sub_class", "any_value"])
    else:
        df = df.rename(columns={"any_value": variable})
    return df
