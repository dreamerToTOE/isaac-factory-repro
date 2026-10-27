"""Exercise the actual runner's immediate pre-integration guard without physics."""
import unittest
try:
    import mujoco
except ImportError:
    mujoco = None


@unittest.skipIf(mujoco is None, 'MuJoCo 3.5 required for runner guard tests')
class RunnerGuardTest(unittest.TestCase):
    def test_valid_state_reaches_step_stub_and_each_tampered_state_does_not(self):
        import numpy as np
        from scripts.g4s2a_gripper_v1_model import NativeGripperEnv
        from scripts.g4s2a_gripper_v1_diagnostics import initial
        from scripts.g4s2a_gripper_v1_r1_state import snapshot, StateIntegrityError
        from scripts.g4s2a_gripper_v1_r1_diagnostics import one_step

        class Ledger:
            def __init__(self): self.requests = 0
            def record_request(self, *args, **kwargs): self.requests += 1
            def record_exception(self, *args, **kwargs): pass

        def corrupt(env, kind):
            if kind == 'arm_qpos': env.data.qpos[env.arm_qpos_ids[0]] += .001
            elif kind == 'finger_qpos': env.data.qpos[env.finger_qpos_ids[0]] += .001
            elif kind == 'peg_position': env.data.qpos[env.peg_qpos_adr] += .001
            elif kind == 'peg_quaternion': env.data.qpos[env.peg_qpos_adr + 3] += .001
            elif kind == 'qvel': env.data.qvel[0] += .001
            elif kind == 'time': env.data.time += .001
            elif kind == 'applied_force': env.data.qfrc_applied[0] += .001
            elif kind == 'warmstart': env.data.qacc_warmstart[0] += .001

        kinds = ('valid', 'arm_qpos', 'finger_qpos', 'peg_position', 'peg_quaternion',
                 'qvel', 'time', 'applied_force', 'warmstart')
        original = mujoco.mj_step
        try:
            for name in ('I2', 'I3'):
                for kind in kinds:
                    with self.subTest(initial=name, corruption=kind):
                        env = NativeGripperEnv(); initial(env, name)
                        expected = snapshot(mujoco, env.model, env.data)
                        if kind != 'valid': corrupt(env, kind)
                        called = []
                        def forbidden(*args):
                            called.append(True)
                            raise RuntimeError('stub: no physical integration')
                        mujoco.mj_step = forbidden
                        ledger = Ledger()
                        if kind == 'valid':
                            with self.assertRaisesRegex(RuntimeError, 'stub: no physical integration'):
                                one_step(env, np.zeros(7), .00498, ledger, 1, expected)
                            self.assertEqual(len(called), 1)
                            self.assertEqual(ledger.requests, 1)
                        else:
                            with self.assertRaises(StateIntegrityError) as caught:
                                one_step(env, np.zeros(7), .00498, ledger, 1, expected)
                            self.assertEqual(caught.exception.boundary, 'immediately_before_mj_step')
                            self.assertFalse(called)
                            self.assertEqual(ledger.requests, 0)
        finally:
            mujoco.mj_step = original

if __name__ == '__main__':
    unittest.main()
