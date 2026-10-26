# G4-S1B adapted Panda PegInsert: provenance and notices

This is an **adapted zero-shot Sim2Sim engineering environment**. It is not an
exact reconstruction of NVIDIA Isaac Factory, its USD meshes, PhysX cooked SDF,
body inertia, grasp geometry or contact solver. The original G4-S1A exact-asset
reconstruction NO-GO remains valid. No Factory USD, mesh or derived asset is
contained in this directory.

## Panda source and modifications

The robot starts from the **Franka Emika Panda**, never FR3, in
[MuJoCo Menagerie, pinned commit 8161bba264d7fa7c99ca301e91e7fb44737676ad](https://github.com/google-deepmind/mujoco_menagerie/tree/8161bba264d7fa7c99ca301e91e7fb44737676ad/franka_emika_panda).
The original Panda README identifies its derivation from Franka ROS and releases
the model under Apache-2.0. The complete, byte-identical original model license
is retained here as [LICENSE_PANDA](LICENSE_PANDA). No NOTICE file was present in
the pinned repository root or Panda model directory.

The S1B builder generates an adapted MJCF in memory, using the external pinned
Panda asset directory. Changes from the source model include the registered
physics timestep and solver, explicit seven-axis torque actuation, separate
gripper control, a mapped TCP frame, gravity policy and collision filtering,
and independently authored assembly bodies. The implementation and evidence
record the actual complete changes and configuration; unchanged visual and
robot collision meshes remain external dependencies. Keep this modification
notice and the Apache-2.0 license when distributing a generated Panda model or
any included Panda assets. This project's MIT license does not replace the
Panda license.

## Independently authored assembly geometry

The peg, bore walls/floor and support geometry are independently generated from
analytic dimensions and formulas. No original Factory vertices, triangles,
mesh decomposition or cooked SDF is copied, exported or converted. These newly
authored implementation portions use the repository's [MIT license](../../LICENSE).
The primary target dimensions are peg diameter **0.007986 m** and hole diameter
**0.0081 m**, giving a nominal radial clearance of **57 μm**. This dimension choice
is an adapted protocol, not a claim that the new geometry equals Factory's raw
mesh or effective PhysX collision surface. A convex-wall approximation must
report its geometric error relative to that clearance.

The peg is an independent free rigid body. The bore opening must remain open
under collision; one solid convex hull is not a valid bore representation. No
peg-to-gripper weld is permitted in the physical evaluation configuration.

## Controller/source attribution

The observation, action, task-space PD, dynamically consistent nullspace and
geometric success equations follow the frozen Isaac Lab Factory source at
[commit 90b79bb2d44feb8d833f260f2bf37da3487180ba](https://github.com/isaac-sim/IsaacLab/tree/90b79bb2d44feb8d833f260f2bf37da3487180ba/source/isaaclab_tasks/isaaclab_tasks/direct/factory).
Original source copyright is Copyright (c) 2022–2025, The Isaac Lab Project
Developers. All rights reserved. Its complete BSD-3-Clause terms are retained
in [LICENSE_ISAACLAB_BSD3](LICENSE_ISAACLAB_BSD3). The S1B implementation is a
MuJoCo/NumPy adaptation and is not the original Isaac implementation. Source
ports must retain the original copyright/license attribution; this notice
does not grant any rights to separately licensed NVIDIA Factory assets.

## Runtime dependency

Use **MuJoCo 3.5.0**. MuJoCo is Apache-2.0 and remains installed externally, not
vendor-copied here. Model compilation, source/asset hashes, geometry dimensions
and no-policy validation are recorded in `evidence/g4_s/s1b/`. No policy
checkpoint is loaded by the model builder and no policy-driven episode belongs
to G4-S1B.
