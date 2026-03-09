import cudf
import numpy as np
from pathlib import Path
from typing import Tuple, Iterator
import hashlib
import cupy as cp

# Criteo dataset column names
# 1 label + 13 integer features + 26 categorical features
LABEL_COL = "label"
INT_COLS = [f"I{i}" for i in range(1, 14)]
CAT_COLS = [f"C{i}" for i in range(1, 27)]
ALL_COLS = [LABEL_COL] + INT_COLS + CAT_COLS

# Embedding dimension formula from the DCN paper
# embed_dim = 6 * (cardinality ^ 0.25)
MIN_EMBED_DIM = 4
MAX_EMBED_DIM = 16


def get_embed_dim(cardinality: int) -> int:
    """Calculate embedding dimension based on feature cardinality."""
    dim = int(6 * (cardinality ** 0.25))
    return max(MIN_EMBED_DIM, min(MAX_EMBED_DIM, dim))


def read_criteo_chunks(
    filepath: str,
    chunk_size: int = 100_000
) -> Iterator[cudf.DataFrame]:
    """
    Read Criteo data file in chunks using cuDF.
    Handles both .gz compressed and uncompressed TSV files.
    """
    filepath = Path(filepath)
    
    # Read in chunks using pandas first, then convert to cuDF
    # cuDF doesn't support chunked reading of gz files directly
    import pandas as pd
    
    reader = pd.read_csv(
        filepath,
        sep="\t",
        header=None,
        names=ALL_COLS,
        chunksize=chunk_size,
        dtype=str  # read everything as string initially
    )
    
    for chunk in reader:
        yield cudf.from_pandas(chunk)


def preprocess_chunk(
    chunk: cudf.DataFrame,
    cat_encoders: dict = None
) -> Tuple[cudf.DataFrame, dict]:
    """
    Preprocess a single chunk:
    - Fill missing values
    - Log-normalize integer features
    - Encode categorical features as integers
    
    Returns processed chunk and updated encoders.
    """
    if cat_encoders is None:
        cat_encoders = {col: {"<unknown>": 0} for col in CAT_COLS}

    # --- Integer features ---
    for col in INT_COLS:
        # Fill missing with 0
        chunk[col] = chunk[col].fillna("0")
        # Convert to float
        chunk[col] = chunk[col].astype("float32")
        # Log transform: log(1 + x) as per DCN paper
        chunk[col] = cudf.Series(cp.log1p(chunk[col].abs().values))

    # --- Categorical features ---
    for col in CAT_COLS:
        # Fill missing with unknown token
        chunk[col] = chunk[col].fillna("<unknown>")
        
        # Assign integer index to each unique value
        unique_vals = chunk[col].unique().to_pandas()
        for val in unique_vals:
            if val not in cat_encoders[col]:
                cat_encoders[col][val] = len(cat_encoders[col])
        
        # Map values to indices
        mapping = cat_encoders[col]
        chunk[col] = chunk[col].to_pandas().map(
            lambda x: mapping.get(x, 0)
        )
        chunk[col] = cudf.Series(chunk[col].values).astype("int32")

    # --- Label ---
    chunk[LABEL_COL] = chunk[LABEL_COL].astype("float32")

    return chunk, cat_encoders


def get_cardinalities(cat_encoders: dict) -> dict:
    """Return the vocabulary size for each categorical feature."""
    return {col: len(encoder) for col, encoder in cat_encoders.items()}