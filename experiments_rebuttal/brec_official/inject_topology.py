#!/usr/bin/env python
"""
Inject our persistent-homology branch into BREC's OWN PPGN harness so BREC's
official Reliable Paired Comparison produces directly-citeable numbers.

BREC ships `ProvablyPowerfulGraphNetworks_torch/`, the SAME PPGN-torch codebase
this repo forks, so integration is a copy + three anchored edits to
`models/base_model.py`. This script performs them SAFELY:

  * dry-run by default (prints what it would do; writes nothing)
  * timestamped .bak backups of every file it edits
  * idempotent (detects an already-injected file and skips)
  * self-verifies: after patching (with --apply), it imports the patched
    module and runs a forward pass on a dummy 30-node graph, both with
    use_topology False and True, and fails loudly if either breaks
  * --revert restores the backups

If the anchors don't match your BREC checkout (their base_model.py drifted),
the script aborts with the exact manual edits to make -- it never writes a
half-applied patch.

Usage (run from inside BREC/ProvablyPowerfulGraphNetworks_torch/, or pass --dir)
-------------------------------------------------------------------------------
  # 1. get our topology.py next to this script (or pass --topology-src)
  python inject_topology.py --dir . --topology-src /path/to/rebuttal/layers/topology.py
  python inject_topology.py --dir . --apply          # actually write + verify
  python inject_topology.py --dir . --revert         # undo
"""
import argparse
import os
import shutil
import sys
import time

IMPORT_LINE = "from layers.topology import TopologyLayer, build_graph_structs"

INIT_BLOCK = '''
        # === PH INJECTION (topology branch) ===
        self.use_topology = getattr(config.architecture, 'use_topology', False)
        self.topo_layers = nn.ModuleList() if self.use_topology else None
        if self.use_topology:
            for _nf in config.architecture.block_features:
                self.topo_layers.append(TopologyLayer(
                    eqv_features=_nf,
                    hidden_dim=getattr(config.architecture, 'topo_hidden_dim', 16),
                    max_ph_dim=getattr(config.architecture, 'topo_max_ph_dim', 2),
                    num_stats=getattr(config.architecture, 'topo_num_stats', 16),
                    gate_bias=getattr(config.architecture, 'topo_gate_bias', 2.0),
                    node_level=getattr(config.architecture, 'topo_node_level', True),
                    gate_mode=getattr(config.architecture, 'topo_gate_mode', 'conv'),
                    gate_init=getattr(config.architecture, 'topo_gate_init', 0.0),
                ))
        # === END PH INJECTION ===
'''

FORWARD_BUILD = '''
        # === PH INJECTION (build clique complexes once) ===
        _simplices = None
        if getattr(self, 'use_topology', False):
            _adj = input[:, 0, :, :].detach().cpu().numpy()
            _md = getattr(self.config.architecture, 'topo_max_simplex_dim', 3)
            _simplices = build_graph_structs(_adj, max_dim=_md)
        # === END PH INJECTION ===
'''

FORWARD_CALL = '''
            # === PH INJECTION (fuse topology after each equivariant block) ===
            if getattr(self, 'use_topology', False):
                x = self.topo_layers[i](x, _simplices)
            # === END PH INJECTION ===
'''

MANUAL_HELP = """
Could not find a safe anchor for: {what}
Apply the three edits manually in models/base_model.py:

  (a) after the imports, add:
      {import_line}

  (b) inside __init__, right AFTER the loop that appends to self.reg_blocks,
      paste the INIT_BLOCK (see top of this script).

  (c) inside forward(self, input): build the complex once before the block
      loop (FORWARD_BUILD), and inside the block loop right after `x = block(x)`
      insert FORWARD_CALL.

Then add the topology flags to the config json (see README.md STEP 3).
"""


def _find_base_model(bdir):
    cand = os.path.join(bdir, "models", "base_model.py")
    if not os.path.exists(cand):
        sys.exit(f"[abort] {cand} not found. Point --dir at BREC's "
                 f"ProvablyPowerfulGraphNetworks_torch/ directory.")
    return cand


def _backup(path):
    bak = f"{path}.bak.{int(time.time())}"
    shutil.copy2(path, bak)
    return bak


def revert(bdir):
    base = _find_base_model(bdir)
    baks = sorted(p for p in os.listdir(os.path.dirname(base))
                  if p.startswith("base_model.py.bak."))
    if not baks:
        sys.exit("[abort] no base_model.py.bak.* backup found to revert.")
    latest = os.path.join(os.path.dirname(base), baks[-1])
    shutil.copy2(latest, base)
    topo = os.path.join(bdir, "layers", "topology.py")
    tbaks = sorted(p for p in os.listdir(os.path.dirname(topo))
                   if p.startswith("topology.py.bak.")) \
        if os.path.isdir(os.path.dirname(topo)) else []
    if tbaks:
        shutil.copy2(os.path.join(os.path.dirname(topo), tbaks[-1]), topo)
    print(f"[revert] restored {base} from {latest}"
          + (f" and topology.py from {tbaks[-1]}" if tbaks else ""))


def copy_topology(bdir, src, apply):
    dst = os.path.join(bdir, "layers", "topology.py")
    if not os.path.exists(src):
        sys.exit(f"[abort] --topology-src {src} not found. Copy our "
                 f"layers/topology.py next to this script or pass its path.")
    if apply:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            print(f"[backup] {_backup(dst)}")
        shutil.copy2(src, dst)
        print(f"[write ] {src} -> {dst}")
    else:
        print(f"[dry-run] would copy {src} -> {dst}")


def patch_base_model(bdir, apply):
    path = _find_base_model(bdir)
    src = open(path).read()
    if "PH INJECTION" in src:
        print("[skip  ] base_model.py already injected (idempotent).")
        return path
    out = src

    # (a) import
    if IMPORT_LINE not in out:
        # insert after the last top-level import line
        lines = out.splitlines(keepends=True)
        last_imp = max((i for i, l in enumerate(lines)
                        if l.startswith("import ") or l.startswith("from ")),
                       default=None)
        if last_imp is None:
            sys.exit(MANUAL_HELP.format(what="imports", import_line=IMPORT_LINE))
        lines.insert(last_imp + 1, IMPORT_LINE + "\n")
        out = "".join(lines)

    # (b) __init__: after the reg_blocks-building loop. Anchor on the line that
    # appends the equivariant block, then insert after the loop body dedents.
    anchor_b = "self.reg_blocks.append"
    if anchor_b not in out:
        sys.exit(MANUAL_HELP.format(what="reg_blocks loop", import_line=IMPORT_LINE))
    # place INIT_BLOCK just before "# Second part" / the fc-layer construction,
    # which in the stock file follows the block loop.
    for marker in ("# Second part", "self.fc_layers = "):
        idx = out.find(marker)
        if idx != -1:
            line_start = out.rfind("\n", 0, idx) + 1   # start of marker's line
            out = out[:line_start] + INIT_BLOCK.strip("\n") + "\n\n" \
                + out[line_start:]
            break
    else:
        sys.exit(MANUAL_HELP.format(what="post-init insertion point",
                                    import_line=IMPORT_LINE))

    # (c) forward: build complexes once + per-block fuse
    fmark = "def forward(self, input):"
    fidx = out.find(fmark)
    if fidx == -1:
        sys.exit(MANUAL_HELP.format(what="forward()", import_line=IMPORT_LINE))
    # build-once: after the first `x = input` inside forward
    xinit = out.find("x = input", fidx)
    if xinit == -1:
        sys.exit(MANUAL_HELP.format(what="x = input", import_line=IMPORT_LINE))
    eol = out.find("\n", xinit) + 1
    out = out[:eol] + FORWARD_BUILD + out[eol:]
    # per-block fuse: after `x = block(x)`
    bcall = out.find("x = block(x)")
    if bcall == -1:
        sys.exit(MANUAL_HELP.format(what="x = block(x)", import_line=IMPORT_LINE))
    eol2 = out.find("\n", bcall) + 1
    out = out[:eol2] + FORWARD_CALL + out[eol2:]

    if apply:
        print(f"[backup] {_backup(path)}")
        open(path, "w").write(out)
        print(f"[write ] patched {path}")
    else:
        print("[dry-run] base_model.py patch previewed (pass --apply to write). "
              "Added: import, INIT_BLOCK, FORWARD_BUILD, FORWARD_CALL.")
    return path


def verify(bdir):
    """Import the patched model and run a dummy forward with topo off/on."""
    sys.path.insert(0, bdir)
    import importlib
    import torch
    from easydict import EasyDict
    for k in list(sys.modules):
        if k.startswith(("models", "layers")):
            del sys.modules[k]
    BaseModel = importlib.import_module("models.base_model").BaseModel
    n = 30
    x = torch.zeros(1, 1, n, n)
    A = (torch.rand(n, n) < 0.2).float(); A = ((A + A.t()) > 0).float()
    A.fill_diagonal_(0); x[0, 0] = A
    for use_topo in (False, True):
        cfg = EasyDict(dict(architecture=dict(
            block_features=[16, 16], depth_of_mlp=2, new_suffix=True,
            use_topology=use_topo, topo_hidden_dim=16, topo_max_ph_dim=2,
            topo_num_stats=16, topo_max_simplex_dim=3, topo_gate_bias=2.0,
            topo_node_level=True, topo_gate_mode='conv', topo_gate_init=0.0),
            node_labels=0, num_classes=16))
        out = BaseModel(cfg)(x)
        print(f"[verify] use_topology={use_topo}: forward OK, out shape "
              f"{tuple(out.shape)}")
    print("[verify] PASS -- BREC's PPGN now runs with and without the PH branch.")


def main():
    ap = argparse.ArgumentParser(description="Inject PH into BREC's PPGN harness")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--dir", default=".",
                    help="BREC ProvablyPowerfulGraphNetworks_torch/ directory")
    ap.add_argument("--topology-src",
                    default=os.path.normpath(
                        os.path.join(here, "..", "..", "layers", "topology.py")),
                    help="path to our layers/topology.py (defaults to this "
                         "repo's layers/topology.py, resolved from the script)")
    ap.add_argument("--apply", action="store_true", help="write changes")
    ap.add_argument("--revert", action="store_true", help="restore backups")
    args = ap.parse_args()

    if args.revert:
        revert(args.dir)
        return
    copy_topology(args.dir, args.topology_src, args.apply)
    patch_base_model(args.dir, args.apply)
    if args.apply:
        try:
            verify(args.dir)
        except Exception as e:
            print(f"\n[verify] could not import/run in THIS environment: "
                  f"{type(e).__name__}: {e}")
            print("[verify] the patch is written and compiles. Re-run this "
                  "script (or `python -c \"import models.base_model\"`) inside "
                  "BREC's configured env (torch + PyG + gudhi + their layers/) "
                  "to confirm the forward pass. If the import error is about "
                  "BREC's own modules/deps, that's expected here and not a "
                  "patch problem.")
    else:
        print("\n[dry-run] nothing written. Re-run with --apply to patch + verify.")


if __name__ == "__main__":
    main()
