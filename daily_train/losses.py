import torch 

def gaussian_kernel(a, b):
    """
    Compute the Gaussian kernel between two tensors.

    Args:
    a (Tensor): A tensor of shape (batch1, seq_len1, embed_dim).
    b (Tensor): A tensor of shape (batch2, seq_len2, embed_dim).

    Returns:
    Tensor: The computed Gaussian kernel.
    """
    batch1, seq_len1, embed_dim = a.shape
    batch2, seq_len2, _ = b.shape

    # Reshape to prepare for pairwise distance computation
    a = a.view(batch1, 1, seq_len1, 1, embed_dim)
    b = b.view(1, batch2, 1, seq_len2, embed_dim)

    # Expand to create all pairwise combinations
    a_core = a.expand(batch1, batch2, seq_len1, seq_len2, embed_dim)
    b_core = b.expand(batch1, batch2, seq_len1, seq_len2, embed_dim)

    # Calculate the numerator of the Gaussian kernel
    numerator = (a_core - b_core).pow(2).mean(-1) / embed_dim
    return torch.exp(-numerator)

def MMD_loss(a, b):
    """
    Compute the Maximum Mean Discrepancy (MMD) loss between two tensors.

    Args:
    a (Tensor): A tensor representing the first set of samples (batch, seq_len, embed_dim).
    b (Tensor): A tensor representing the second set of samples (batch, seq_len, embed_dim).

    Returns:
    Tensor: The computed MMD loss.
    """
    return gaussian_kernel(a, a).mean() + gaussian_kernel(b, b).mean() - 2 * gaussian_kernel(a, b).mean()