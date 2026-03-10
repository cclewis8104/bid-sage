import torch
import torch.nn as nn
from typing import Dict, List


class EmbeddingLayer(nn.Module):
    """
    Converts categorical feature indices into dense embedding vectors,
    then concatenates with numerical features to produce the input
    vector x0 for the cross and deep networks.
    """

    def __init__(
        self,
        cardinalities: Dict[str, int],
        embed_dims: Dict[str, int],
        num_numerical: int
    ):
        """
        Args:
            cardinalities: vocab size per categorical feature
                           e.g. {"C1": 22538, "C2": 5897, ...}
            embed_dims: embedding dimension per categorical feature
                        e.g. {"C1": 16, "C2": 14, ...}
            num_numerical: number of numerical features (13 for Criteo)
        """
        super().__init__()

        self.cat_cols = sorted(cardinalities.keys())

        # Create one embedding table per categorical feature
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(
                num_embeddings=cardinalities[col] + 1,  # +1 for unknown
                embedding_dim=embed_dims[col],
                padding_idx=0  # index 0 = unknown/missing
            )
            for col in self.cat_cols
        })

        # Total dimension of x0
        self.embed_output_dim = sum(embed_dims[col] for col in self.cat_cols)
        self.output_dim = self.embed_output_dim + num_numerical

    def forward(
        self,
        numerical: torch.Tensor,
        categorical: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            numerical: float tensor of shape (batch_size, num_numerical)
            categorical: int tensor of shape (batch_size, num_categorical)

        Returns:
            x0: float tensor of shape (batch_size, output_dim)
        """
        # Look up embeddings for each categorical feature
        embed_vecs = [
            self.embeddings[col](categorical[:, i])
            for i, col in enumerate(self.cat_cols)
        ]

        # Concatenate all embeddings
        embed_concat = torch.cat(embed_vecs, dim=1)

        # Concatenate with numerical features to form x0
        x0 = torch.cat([embed_concat, numerical], dim=1)

        return x0

class CrossNetwork(nn.Module):
    """
    Cross Network from the DCN paper.
    
    Each layer applies explicit feature crossing:
        x(l+1) = x0 * x(l)^T * w(l) + b(l) + x(l)
    
    The degree of feature interactions grows with depth:
        Layer 1 -> pairwise interactions
        Layer 2 -> 3-way interactions
        Layer L -> (L+1)-way interactions
    """

    def __init__(self, input_dim: int, num_layers: int):
        """
        Args:
            input_dim: dimension of x0 (output_dim from EmbeddingLayer)
            num_layers: number of cross layers (paper uses 6)
        """
        super().__init__()

        self.num_layers = num_layers

        # Each cross layer has one weight vector and one bias vector
        # both of dimension input_dim
        self.weights = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim, 1))
            for _ in range(num_layers)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim, 1))
            for _ in range(num_layers)
        ])

        # Initialize weights with Xavier uniform
        for w in self.weights:
            nn.init.xavier_uniform_(w)

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x0: input tensor of shape (batch_size, input_dim)

        Returns:
            xl: output tensor of shape (batch_size, input_dim)
        """
        # Reshape for matrix operations: (batch_size, input_dim, 1)
        x0 = x0.unsqueeze(2)
        xl = x0.clone()

        for i in range(self.num_layers):
            # x0 * xl^T: (batch_size, input_dim, input_dim)
            # * w:       (batch_size, input_dim, 1)
            # + b:       (batch_size, input_dim, 1)
            # + xl:      (batch_size, input_dim, 1)
            xl = torch.matmul(x0, xl.transpose(1, 2)) \
                @ self.weights[i] \
                + self.biases[i] \
                + xl

        # Remove the trailing dimension: (batch_size, input_dim)
        return xl.squeeze(2)

class DeepNetwork(nn.Module):
    """
    Standard fully-connected feed-forward network with ReLU activations
    and batch normalization, as described in the DCN paper.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.0
    ):
        """
        Args:
            input_dim: dimension of x0 (same as CrossNetwork input)
            hidden_dims: list of hidden layer sizes
                         e.g. [1024, 1024] for paper's optimal setting
            dropout: dropout rate between layers (0.0 = disabled)
        """
        super().__init__()

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ])
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        self.network = nn.Sequential(*layers)
        self.output_dim = current_dim

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x0: input tensor of shape (batch_size, input_dim)

        Returns:
            h: output tensor of shape (batch_size, output_dim)
        """
        return self.network(x0)

class DCN(nn.Module):
    """
    Deep & Cross Network for CTR prediction.
    
    Architecture:
        1. EmbeddingLayer -> x0
        2. CrossNetwork(x0) -> x_cross  (parallel)
        3. DeepNetwork(x0) -> x_deep    (parallel)
        4. Concat([x_cross, x_deep]) -> logit -> sigmoid -> p
    """

    def __init__(
        self,
        cardinalities: Dict[str, int],
        embed_dims: Dict[str, int],
        num_numerical: int,
        num_cross_layers: int = 6,
        deep_hidden_dims: List[int] = [1024, 1024],
        dropout: float = 0.0
    ):
        """
        Args:
            cardinalities: vocab size per categorical feature
            embed_dims: embedding dimension per categorical feature
            num_numerical: number of numerical features (13 for Criteo)
            num_cross_layers: number of cross layers (paper uses 6)
            deep_hidden_dims: hidden layer sizes for deep network
            dropout: dropout rate for deep network
        """
        super().__init__()

        # Embedding + stacking layer
        self.embedding_layer = EmbeddingLayer(
            cardinalities=cardinalities,
            embed_dims=embed_dims,
            num_numerical=num_numerical
        )

        input_dim = self.embedding_layer.output_dim

        # Cross network
        self.cross_network = CrossNetwork(
            input_dim=input_dim,
            num_layers=num_cross_layers
        )

        # Deep network
        self.deep_network = DeepNetwork(
            input_dim=input_dim,
            hidden_dims=deep_hidden_dims,
            dropout=dropout
        )

        # Combination layer
        # Input: concatenation of cross output (input_dim)
        #        and deep output (deep_hidden_dims[-1])
        combination_dim = input_dim + self.deep_network.output_dim
        self.combination_layer = nn.Linear(combination_dim, 1)

    def forward(
        self,
        numerical: torch.Tensor,
        categorical: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            numerical: float tensor of shape (batch_size, num_numerical)
            categorical: int tensor of shape (batch_size, num_categorical)

        Returns:
            p: predicted click probability (batch_size, 1)
        """
        # Step 1: embed and stack -> x0
        x0 = self.embedding_layer(numerical, categorical)

        # Step 2: cross and deep networks run in parallel
        x_cross = self.cross_network(x0)
        x_deep = self.deep_network(x0)

        # Step 3: concatenate and project to scalar logit
        x_combined = torch.cat([x_cross, x_deep], dim=1)
        logit = self.combination_layer(x_combined)

        # Step 4: sigmoid to get click probability
        return torch.sigmoid(logit)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)