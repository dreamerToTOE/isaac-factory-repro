# G4-S2A-R2 evidence

All new integration is **NO_POLICY_DIAGNOSTIC_ONLY**. No actor or checkpoint is
loaded. The frozen S1B model and all historical S2A/R1 evidence remain unchanged.

- `matrix_v2/preregistered_matrix.json`: active registry, written before new
  integration; fixed 3 initial states × 4 P/C variants, 235 steps each, with
  60 reserved physical-closing steps. Total ceiling 2880, actual 1417.
- `matrix_v2/initials/`: once-only admission attempts and complete available
  integration-state vectors. I1 reuses archived S2A seed1001; I2/I3 are diagnostic
  seeds2101/2102, not new policy evaluations. All three admitted once.
- `matrix_v2/trajectories/`: raw request/result gzip JSONL, physical/controller
  matrices at the same full q, controller inputs and outputs, contacts, interval
  acceleration, native-GUI admission, original live screenshots and termination.
- `matrix_v2/*_process.json`, `summary.json`: exact child exit codes, early-stop
  failures and total accounting. A normal exit does not imply physical stability.
- `numerical_review_v2/`: canonical independent NumPy reconstruction of all
  1357 trajectory steps, admission review, inertia tensor/energy checks, matched
  common-prefix contrasts, late contact-failure witnesses, and scientific plots.
  `numerical_review_v1/` is an earlier successful analysis, retained unchanged.
- `regression_prelaunch_v2/`: passing CPU and static-model admission receipt.
  `regression_prelaunch_v1/` is a preserved development failure: all pure tests
  passed, but an overly strict compiler principal-axis roundoff assertion and
  source changes during the audit failed the aggregate gate. No physics followed
  that failed gate. `regression_final_v1/` records the post-implementation checks.
- `visuals/`: preferred actual-X11 saved-state captures are selected by the final
  image QA manifest. Playback performs zero integration; it is not a rerun of the
  original solver impulses. Original live and earlier playback captures remain
  intact, including oversized/unpinned visual-option overlays. No image pixels
  were generated or edited. Scientific plots are explicitly separate artifacts.
  Exactly two screenshots captured unrelated desktop windows and remain private
  outside Git. `visuals/private_capture_availability.json` preserves their paths
  and hashes; `live_image_qa.json` and `earlier_playback_qa.json` cover all original
  captures. `public_manifest.json` selects14 individually reviewed preferred v3
  screenshots. Numeric traces are complete and public.
- `matrix_v1/`: unexecuted registry retained after a supervision-only amendment;
  see `preregistration_amendment.json`. Scientific settings were unchanged.

The outcome is **CANDIDATE_PARTIAL**. P1 suppresses initial oscillation in all
three registered poses, while C1 alone does not. B/D meet the full-horizon gates
in I1 and I2, but I3_B/I3_D terminate at177/216 steps after transient left-contact
loss and >1mm fingertip/peg penetration. This does not justify R3 or S2B GO.
ROOT_CAUSE_PENDING / SOLUTION_PENDING / VALIDATION_PENDING remain applicable.

`matrix_v*/P0_diagnostic.xml` and `P1_diagnostic.xml` are explicit, separate
diagnostic configurations. P1 is not installed as the default environment or
approved for policy validation. They reference the pinned external Menagerie
assets without copying meshes. The Panda-derived portions retain Apache-2.0
provenance, commit `8161bba264d7fa7c99ca301e91e7fb44737676ad`; the license is
preserved at [`models/g4_s1b/LICENSE_PANDA`](../../../models/g4_s1b/LICENSE_PANDA).
The analytic assembly geometry is authored by this repository. Controller source
port provenance remains in
[`LICENSE_ISAACLAB_BSD3`](../../../models/g4_s1b/LICENSE_ISAACLAB_BSD3).
No original NVIDIA asset or derived original geometry is redistributed.
