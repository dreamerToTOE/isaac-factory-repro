"""Bounded standalone Isaac Sim Vulkan / CUDA PhysX gate (no remote assets)."""
import json
from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0})
try:
    import numpy as np
    import torch
    import omni.physics.tensors as tensors
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid

    world = World(physics_dt=1 / 120, rendering_dt=1 / 60, backend="torch", device="cuda:0")
    context = world.get_physics_context()
    context.enable_gpu_dynamics(True)
    context.set_broadphase_type("GPU")
    world.scene.add_default_ground_plane()
    cube = world.scene.add(DynamicCuboid(prim_path="/World/probe_cube", name="probe_cube",
                                        position=np.array([0., 0., 1.]), scale=np.array([.1, .1, .1]), mass=1.))
    world.reset()
    view = tensors.create_simulation_view("torch")
    view.set_subspace_roots("/")
    rigid = view.create_rigid_body_view("/World/probe_cube")
    start = rigid.get_transforms().clone()
    for _ in range(240):
        world.step(render=False)
    end = rigid.get_transforms().clone()
    result = {"cuda_available": torch.cuda.is_available(), "torch_version": torch.__version__,
              "gpu": torch.cuda.get_device_name(0), "tensor_device": str(end.device),
              "gpu_dynamics": context.is_gpu_dynamics_enabled(), "start": start.tolist(), "end": end.tolist(),
              "finite": bool(torch.isfinite(end).all()), "steps": 240}
    result["passed"] = (result["gpu_dynamics"] and end.is_cuda and result["finite"]
                        and float(start[0, 2]) > .9 and .03 < float(end[0, 2]) < .1)
    print("GPU_PROBE_RESULT=" + json.dumps(result), flush=True)
    if not result["passed"]:
        raise RuntimeError("GPU-backed physics gate failed")
finally:
    app.close()
