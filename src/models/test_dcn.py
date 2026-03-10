import sys
sys.path.append('/home/cclewis8104/bid-sage')

import torch
from src.models.dcn import DCN
from src.data.loader import get_embed_dim, CAT_COLS, INT_COLS

# Simulate cardinalities from our loader output
cardinalities = {
    "C1": 22538, "C2": 5897, "C3": 7687, "C4": 2565,
    "C5": 6094, "C6": 4, "C7": 4502, "C8": 920,
    "C9": 30, "C10": 19493, "C11": 8636, "C12": 12718,
    "C13": 10, "C14": 1125, "C15": 3038, "C16": 44,
    "C17": 4, "C18": 426, "C19": 15, "C20": 23250,
    "C21": 14570, "C22": 21460, "C23": 7225, "C24": 5315,
    "C25": 40, "C26": 32
}

embed_dims = {col: get_embed_dim(card) for col, card in cardinalities.items()}

# Instantiate model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = DCN(
    cardinalities=cardinalities,
    embed_dims=embed_dims,
    num_numerical=len(INT_COLS),
    num_cross_layers=6,
    deep_hidden_dims=[1024, 1024],
    dropout=0.0
).to(device)

print(f"\nTotal trainable parameters: {model.count_parameters():,}")

# Test forward pass with random batch
batch_size = 512
numerical = torch.randn(batch_size, len(INT_COLS)).to(device)
cat_cols_sorted = sorted(cardinalities.keys())
categorical = torch.stack([
    torch.randint(0, cardinalities[col], (batch_size,))
    for col in cat_cols_sorted
], dim=1).to(device)

output = model(numerical, categorical)
print(f"\nInput numerical shape:  {numerical.shape}")
print(f"Input categorical shape: {categorical.shape}")
print(f"Output shape:            {output.shape}")
print(f"Output range:            [{output.min().item():.4f}, {output.max().item():.4f}]")
print(f"\nModel instantiation successful!")