"""Pin learning modules before Kit prepends its bundled archive paths; run upstream unchanged."""
import json
import os
import runpy
import sys
import gymnasium
import numpy
import torch
import torchvision

packages = {"torch": torch, "torchvision": torchvision, "numpy": numpy, "gymnasium": gymnasium}
expected = {"torch": "2.7.0+cu128", "torchvision": "0.22.0+cu128", "numpy": "1.26.4", "gymnasium": "1.2.0"}

def report(stage):
    result = {name: {"version": module.__version__, "path": module.__file__} for name, module in packages.items()}
    assert all(result[name]["version"] == version for name, version in expected.items()), result
    assert all(sys.modules[name] is module for name, module in packages.items())
    print(stage + "=" + json.dumps(result), flush=True)

report("PINNED_PACKAGES")
if len(sys.argv) < 2:
    raise SystemExit("Usage: run_sim.sh SCRIPT.py [arguments]")

# This wrapper adds evidence only; the official launcher and training code run unchanged.
from isaaclab.app import AppLauncher
original_init = AppLauncher.__init__
def audited_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    report("PACKAGES_AFTER_APP_LAUNCH")
AppLauncher.__init__ = audited_init

# Optional read-only G2 logging; the target remains the official training script.
if os.environ.get("FACTORY_G2_AUDIT_OUTPUT"):
    from g2_training_observer import install
    install(AppLauncher)

target = sys.argv[1]
sys.argv = sys.argv[1:]
if target == "-c":
    exec(sys.argv[1], {"__name__": "__main__"})
else:
    runpy.run_path(target, run_name="__main__")
