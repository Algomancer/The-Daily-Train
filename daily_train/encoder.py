# Not reusing the model.py transformer for the encoder, mainly so I don't break any pretrained decoder loading with 'random' modification.
# Lil code duplication never hurt anyone. 
# Narrator: It did.
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_rope_cache(seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000) -> torch.Tensor:
    """
    This function creates a cache for the Rotary Position Embeddings (RoPE).

    Parameters
    ----------
    seq_len : int
        The length of the sequence for which the position embeddings are calculated.

    n_elem : int
        The number of elements in the sequence.

    dtype : torch.dtype
        The data type of the tensor.

    device : torch.device
        The device on which the tensor will be allocated.

    base : int, optional (default=10000)
        The base number used in the calculation of the position embeddings.

    Returns
    -------
    cache : torch.Tensor
        The tensor containing the position embeddings.

    Notes
    -----
    The cache is a matrix, with each row corresponding to a position in the sequence and each column corresponding to a dimension in the embedding space.
    The values in the matrix are computed based on the frequency associated with each position and dimension, and transformed using cosine and sine functions.
    """

    # Compute the reciprocal of the sequence length, raised to the power of each position in the sequence.
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))

    # Create a sequence of position indexes `[0, 1, ..., seq_len - 1]`.
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

    # Compute the outer product of the position index and theta.
    idx_theta = torch.outer(seq_idx, theta).float()

    # Create the cache by applying cosine and sine functions to the outer product.
    cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

    # Adjust the data type of the cache to match the input data type.
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        cache = cache.half()
    return cache


def apply_rope(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    """
    This function applies the Rotary Position Embeddings to the input tensor.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor to which the Rotary Position Embeddings will be applied.

    rope_cache : torch.Tensor
        The tensor containing the Rotary Position Embeddings.

    Returns
    -------
    x_out2 : torch.Tensor
        The tensor resulting from the application of the Rotary Position Embeddings.

    Notes
    -----
    The function works by first adjusting the shapes of the input tensor and the Rotary Position Embeddings to match.
    Then it applies the Rotary Position Embeddings by performing complex multiplication.
    Finally, it flattens and transposes the output tensor to match the original input tensor shape.
    """

    # Transpose the tensor to match the shape of the Rotary Position Embeddings.
    x = x.transpose(1, 2)

    # Truncate the cache to match the size of the input tensor.
    T = x.size(1)
    rope_cache = rope_cache[:T]

    # Adjust the data type of the input tensor to match the cache.
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    rope_cache = rope_cache.view(1, xshaped.size(1), 1, xshaped.size(3), 2)

    # Apply the Rotary Position Embeddings by performing complex multiplication.
    # multiplication.
    x_out2 = torch.stack(
        [xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
         xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ], -1)

    # Flatten and transpose the tensor to match the original input tensor shape.
    x_out2 = x_out2.flatten(3)
    return x_out2.transpose(1, 2).type_as(x)



class SelfAttention(nn.Module):
    """
    Implementation of a self-attention mechanism, with optional causality and rotary position embedding.
    """
    def __init__(self, embed_dim, heads, block_size, causal=False, rope=False) -> None:
        """
        Constructor for the SelfAttention class.

        Parameters:
        embed_dim (int): The embedding dimension.
        num_heads (int): The number of attention heads.
        block_size (int): The length of the sequence.
        causal (bool): If True, uses causal attention. Useful for autoregressive modeling.
        rope (bool): If True, uses rotary position embedding for positional encoding.
        """
        super().__init__()
        assert embed_dim % heads == 0

        # Key, query, value projections for all heads, but in a batch
        self.compute_qkv = nn.Linear(embed_dim, 3 * embed_dim)

        # Output projection
        self.compute_out = nn.Linear(embed_dim, embed_dim)

        # Store parameters
        self.embed_dim = embed_dim
        self.num_heads = heads
        self.block_size = block_size
        self.casual = causal
        self.rope = rope
        self.rope_cache = None
        self.head_size = embed_dim // heads


    def forward(self, x, mask=None):
        """
        The forward pass of the self-attention mechanism.

        Parameters:
        x (torch.Tensor): The input tensor.
        mask (torch.Tensor): An optional mask tensor.

        Returns:
        y (torch.Tensor): The output tensor after the self-attention mechanism.
        """
        # Unpack the shape of the input tensor
        batch, seq_len, feature_dim = x.shape

        # Compute query, key, values for all heads in batch and move head forward to be the batch dimension
        query, key, value = self.compute_qkv(x).split(self.embed_dim, dim=2)
        head_dim = self.embed_dim // self.num_heads
        key = key.view(batch, seq_len, self.num_heads, head_dim).transpose(1, 2)
        query = query.view(batch, seq_len, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(batch, seq_len, self.num_heads, head_dim).transpose(1, 2)

        # If using rotary position embedding and the cache is not yet built, build it
        query, key = self.resolve_rope(x, query, key)

        # Apply efficient attention using Flash Attention CUDA kernels
        y = F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=self.casual
        )

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)

        # Output projection
        y = self.compute_out(y)
        return y

    def resolve_rope(self, x, query, key):
        if self.rope and self.rope_cache is None:
            self.rope_cache = build_rope_cache(
                seq_len=self.block_size,
                n_elem=self.embed_dim // self.num_heads, 
                dtype=x.dtype,
                device=x.device,
            )

        # If using rotary position embedding, apply it
        if self.rope:
            query = apply_rope(query, self.rope_cache)
            key = apply_rope(key, self.rope_cache)
        return query,key


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization module. This is a variant of layer normalization that normalizes using the root mean square.

    Args:
        dim (int): The number of dimensions for the layer normalization.
        scale (bool): Whether to include scale in the layer normalization. 
        eps (float, optional): Small number for numerical stability. Default is 1e-5.
    """
    def __init__(self, dim, scale=True, eps=1e-5):
        super().__init__()  # Inherit methods and properties from nn.Module.
        # Initialize the scale parameter with ones if scale is True, else set it to None.
        self.scale = nn.Parameter(torch.ones(dim)) if scale else scale
        self.eps = eps  # Small number for numerical stability.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the RMS Layer Normalization. Normalizes the features of the input tensor using the root mean square.

        Args:
            x (torch.Tensor): Input tensor that needs to be normalized.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        # Calculate the mean of the squares of the input tensor.
        norm_x = torch.mean(x * x, dim=-1, keepdim=True)
        # Normalize the input tensor using the root mean square.
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        if self.scale is not None:  # If scale is not None.
            # Multiply the normalized tensor by the scale.
            return self.scale * x_normed
        return x_normed  # If scale is None, return the normalized tensor.

class LLaMAMLP(nn.Module):
    """
    An implementation of a LLaMAs Multi-Layer Perceptron (MLP) in PyTorch. This MLP uses SiLU activation function and
    elementwise multiplication between the outputs of two hidden layers. This is the MLP used in the LLaMA implimentation.

    Parameters
    ----------
    input_dim : int
        The dimensionality of the input data.
    hidden_ratio : float
        The ratio to compute the hidden dimension size.
    block_size : int
        The size of the block for calculating n_hidden.
    """

    def __init__(self, input_dim: int, hidden_ratio: float = 4.0, block_size: int = 256) -> None:
        super().__init__()
        hidden_dim = hidden_ratio * input_dim
        n_hidden = int(2 * hidden_dim / 3)
        # Ensure n_hidden is a multiple of block size
        n_hidden = ((n_hidden - 1) // block_size) * block_size + block_size

        # First and second hidden layers
        self.hidden_layer_1 = nn.Linear(input_dim, n_hidden, bias=False)
        self.hidden_layer_2 = nn.Linear(input_dim, n_hidden, bias=False)

        # Output projection layer
        self.output_layer = nn.Linear(n_hidden, input_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Parameters
        ----------
        x : torch.Tensor
            The input data.

        Returns
        -------
        output : torch.Tensor
            The output data.
        """
        x = F.silu(self.hidden_layer_1(x)) * self.hidden_layer_2(x)
        x = self.output_layer(x)
        return x


class Block(nn.Module):
    def __init__(self, embed_dim, heads, block_size, causal=False, rope=False, normalisation=RMSNorm) -> None:
        super().__init__()
        self.rms_1 = normalisation(embed_dim)
        self.attn = SelfAttention(embed_dim, heads, block_size, causal, rope)
        self.rms_2 = normalisation(embed_dim)
        self.mlp = LLaMAMLP(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.rms_1(x))
        x = x + self.mlp(self.rms_2(x))
        return x


class TransformerBase(nn.Module):
    """
    Transformer with arbitrary input and output dimensions. Assumes input embed is done externally.

    """

    def __init__(
        self, embed_dim, num_heads, block_size, layers, causal=False, rope=False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.blocks = nn.ModuleList(
            [
                Block(embed_dim, num_heads, block_size, causal, rope)
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x
