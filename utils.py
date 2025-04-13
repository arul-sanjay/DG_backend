
import pandas as pd
import numpy as np
import logging
from typing import List, Union, Any
import io

logger = logging.getLogger(__name__)

def load_dataset(file: Union[io.BytesIO, Any]) -> pd.DataFrame:
    """Load dataset from file."""
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            return pd.read_excel(file)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing file: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def sample_dataframe(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """Sample the dataframe if it exceeds the sample size."""
    if len(df) > sample_size:
        logger.info(f"Sampling dataset from {len(df)} to {sample_size} rows")
        return df.sample(n=sample_size, random_state=42)
    return df

def check_unique_columns(df: pd.DataFrame) -> List[str]:
    """Identify columns with unique values, potential identifiers."""
    unique_cols = [col for col in df.columns if df[col].nunique() == len(df)]
    if unique_cols:
        logger.info(f"Found unique columns: {unique_cols}")
    return unique_cols

def convert_to_json_serializable(obj):
    """Convert NumPy types to JSON serializable Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj
