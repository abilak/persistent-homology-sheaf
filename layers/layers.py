import torch


def diag_offdiag_maxpool(input):
    N = input.shape[-1]

    max_diag = torch.max(torch.diagonal(input, dim1=-2, dim2=-1), dim=2)[0]  # BxS

    # with torch.no_grad():
    max_val = torch.max(max_diag)
    min_val = torch.max(-1 * input)
    val = torch.abs(torch.add(max_val, min_val))

    min_mat = torch.mul(val, torch.eye(N, device=input.device)).view(1, 1, N, N)

    max_offdiag = torch.max(torch.max(input - min_mat, dim=3)[0], dim=2)[0]  # BxS

    return torch.cat((max_diag, max_offdiag), dim=1)  # output Bx2S


def diag_offdiag_maxpool_masked(input, node_mask):
    """diag_offdiag_maxpool restricted to real nodes (padded batching).
    input (B, S, n, n), node_mask (B, n) bool. For graphs with >= 2 real nodes
    this equals diag_offdiag_maxpool on the unpadded graph exactly (the
    original's diagonal shift never wins the off-diagonal max there). For
    single-node graphs the original returns x_00 minus a batch-dependent shift;
    we use the same formula over the real entries of the batch."""
    B, S, n, _ = input.shape
    neg = torch.finfo(input.dtype).min
    diag = torch.diagonal(input, dim1=-2, dim2=-1)                    # B x S x n
    dm = node_mask[:, None, :]
    # torch.max(dim) (not amax): same first-maximum tie routing of the gradient
    # as the original, and padding sits after the real nodes, so ties among
    # real entries resolve identically.
    max_diag = torch.max(torch.where(dm, diag, torch.full_like(diag, neg)), dim=2)[0]   # B x S
    pair = node_mask[:, :, None] & node_mask[:, None, :]
    eye = torch.eye(n, dtype=torch.bool, device=input.device)
    off = (pair & ~eye)[:, None]                                       # B x 1 x n x n
    max_off = torch.max(torch.max(torch.where(off, input, torch.full_like(input, neg)),
                                  dim=3)[0], dim=2)[0]
    # single-node fallback, mirroring the original shift
    real = pair[:, None].expand_as(input)
    max_val = max_diag.max()
    min_val = torch.where(real, -input, torch.full_like(input, neg)).max()
    val = torch.abs(max_val + min_val)
    n_real = node_mask.sum(1)[:, None]
    max_off = torch.where(n_real >= 2, max_off, max_diag - val)
    return torch.cat((max_diag, max_off), dim=1)
