import sys
sys.path.append('/home/cclewis8104/bid-sage')

from src.data.loader import (
    read_criteo_chunks,
    preprocess_chunk,
    get_cardinalities,
    INT_COLS,
    CAT_COLS,
    LABEL_COL
)

DATA_PATH = "/home/cclewis8104/bid-sage/src/data/day_2.gz"

print("Reading first chunk...")
chunks = read_criteo_chunks(DATA_PATH, chunk_size=100_000)
raw_chunk = next(chunks)
print(f"Raw chunk shape: {raw_chunk.shape}")
print(f"\nSample raw data:")
print(raw_chunk.head(3).to_pandas().to_string())

print("\nPreprocessing chunk...")
processed_chunk, cat_encoders = preprocess_chunk(raw_chunk)

print(f"\nProcessed chunk shape: {processed_chunk.shape}")
print(f"\nSample processed integer features:")
print(processed_chunk[INT_COLS].head(3).to_pandas().to_string())
print(f"\nSample processed categorical features:")
print(processed_chunk[CAT_COLS].head(3).to_pandas().to_string())
print(f"\nLabel distribution:")
print(processed_chunk[LABEL_COL].value_counts().to_pandas())

cardinalities = get_cardinalities(cat_encoders)
print(f"\nCardinalities (vocabulary sizes):")
for col, card in cardinalities.items():
    print(f"  {col}: {card}")