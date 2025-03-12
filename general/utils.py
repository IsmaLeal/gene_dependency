import os
import logging
import pandas as pd


def clean_col_names(col: str) -> str:
    """
    Extracts gene names from a label formatted as 'GeneName (GeneID)'.

    Parameters
    ----------
    col : str
        Column label containing gene name and ID.

    Returns
    -------
    str
        Extracted gene name.
    """
    return col.split(" (")[0]


def safe_read_csv(filepath, **kwargs):
    """
    Reads a CSV file safely, returning an empty DataFrame if the file is missing.

    Parameters
    ----------
    filepath : str
        The path of the file to read as a Pandas DataFrame object.

    Returns
    -------
    pd.DataFrame
        DataFrame contained on `filepath`, or empty DataFrame if file is missing.
    """
    if not os.path.exists(filepath):
        logging.warning(f"File not found: {filepath}")
        return pd.DataFrame()
    return pd.read_csv(filepath, **kwargs)
