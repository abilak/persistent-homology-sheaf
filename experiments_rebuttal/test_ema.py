"""WeightEMA: exact averaging recursion, exact swap-in / restore, off by default."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from trainers.trainer import WeightEMA

torch.manual_seed(0)
net = torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.ReLU(), torch.nn.Linear(7, 2)).double()
ema = WeightEMA(net, 0.99)
ref = [p.detach().clone() for p in net.parameters()]
opt = torch.optim.Adam(net.parameters(), lr=1e-2)
for t in range(1, 301):
    opt.zero_grad(); net(torch.randn(4, 5, dtype=torch.double)).pow(2).sum().backward(); opt.step()
    ema.update()
    d = min(0.99, (1 + t) / (10 + t))
    ref = [d * r + (1 - d) * p.detach() for r, p in zip(ref, net.parameters())]
err = max((s - r).abs().max().item() for s, r in zip(ema.shadow, ref))
raw = [p.detach().clone() for p in net.parameters()]
x = torch.randn(3, 5, dtype=torch.double)
with ema.applied():
    inside = max((p - s).abs().max().item() for p, s in zip(net.parameters(), ema.shadow))
    y_ema = net(x)
restored = max((p - r).abs().max().item() for p, r in zip(net.parameters(), raw))
# a fresh net loaded with the shadow gives the same output
net2 = torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.ReLU(), torch.nn.Linear(7, 2)).double()
with torch.no_grad():
    for p, s in zip(net2.parameters(), ema.shadow):
        p.copy_(s)
out_err = (net2(x) - y_ema).abs().max().item()
ok = err < 1e-12 and inside == 0 and restored == 0 and out_err < 1e-12
print(f"recursion error {err:.1e}, weights inside applied() == shadow: {inside == 0}, "
      f"restored exactly: {restored == 0}, output with averaged weights {out_err:.1e}")
print("PASS" if ok else "FAIL")
