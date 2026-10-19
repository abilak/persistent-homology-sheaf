import torch
import torch.nn as nn

class RegularBlock(nn.Module):
    def __init__(self, config, in_f, out_f):
        super().__init__()
        self.mlp1 = MlpBlock(in_f, out_f, config.architecture.depth_of_mlp)
        self.mlp2 = MlpBlock(in_f, out_f, config.architecture.depth_of_mlp)
        self.skip = SkipConnection(in_f + out_f, out_f)

    def forward(self, inputs, mask2d=None):
        mlp1 = self.mlp1(inputs)
        mlp2 = self.mlp2(inputs)
        if(mask2d is not None):
            mlp1 = mlp1 * mask2d
            mlp2 = mlp2 * mask2d
        mult = torch.matmul(mlp1, mlp2)
        return self.skip(inputs, mult)

class MlpBlock(nn.Module):
    def __init__(self, in_f, out_f, depth_of_mlp, act=nn.functional.relu):
        super().__init__()
        self.activation = act
        self.convs = nn.ModuleList()
        for i in range(depth_of_mlp):
            self.convs.append(nn.Conv2d(in_f, out_f, kernel_size=1, padding=0, bias=True))
            init_w(self.convs[-1])
            in_f = out_f

    def forward(self, x):
        for conv in self.convs:
            x = self.activation(conv(x))
        return x

class SkipConnection(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.conv = nn.Conv2d(in_f, out_f, kernel_size=1, padding=0, bias=True)
        init_w(self.conv)

    def forward(self, in1, in2):
        return self.conv(torch.cat((in1, in2), dim=1))

class FullyConnected(nn.Module):
    def __init__(self, in_f, out_f, act=nn.functional.relu):
        super().__init__()
        self.fc = nn.Linear(in_f, out_f)
        init_w(self.fc)
        self.activation = act

    def forward(self, x):
        out = self.fc(x)
        if(self.activation is not None):
            out = self.activation(out)
        return out

def init_w(layer):
    nn.init.xavier_uniform_(layer.weight)
    if(layer.bias is not None):
        nn.init.zeros_(layer.bias)

def diag_offdiag_maxpool(input):
    N = input.shape[-1]
    mdiag = torch.max(torch.diagonal(input, dim1=-2, dim2=-1), dim=2)[0]  # BxS

    max_val = torch.max(mdiag)
    min_val = torch.max(-1 * input)
    val = torch.abs(torch.add(max_val, min_val))
    min_mat = torch.mul(val, torch.eye(N, device=input.device)).view(1, 1, N, N)

    moff = torch.max(torch.max(input - min_mat, dim=3)[0], dim=2)[0]  # BxS
    return torch.cat((mdiag, moff), dim=1)  # Bx2S

def diag_offdiag_maxpool_masked(input, nmask):
    B, S, n, _ = input.shape
    neg = torch.finfo(input.dtype).min
    diag = torch.diagonal(input, dim1=-2, dim2=-1)
    dm = nmask[:, None, :]
    mdiag = torch.max(torch.where(dm, diag, torch.full_like(diag, neg)), dim=2)[0]
    pair = nmask[:, :, None] & nmask[:, None, :]
    eye = torch.eye(n, dtype=torch.bool, device=input.device)
    off = (pair & ~eye)[:, None]
    max_off = torch.max(torch.max(torch.where(off, input, torch.full_like(input, neg)), dim=3)[0], dim=2)[0]

    # single node graphs
    real = pair[:, None].expand_as(input)
    max_val = mdiag.max()
    min_val = torch.where(real, -input, torch.full_like(input, neg)).max()
    val = torch.abs(max_val + min_val)
    n_real = nmask.sum(1)[:, None]
    max_off = torch.where(n_real >= 2, max_off, mdiag - val)
    return torch.cat((mdiag, max_off), dim=1)
