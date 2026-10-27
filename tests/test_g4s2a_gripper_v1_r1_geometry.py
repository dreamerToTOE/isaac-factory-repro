"""Static geometry check: model load/forward permitted, integration forbidden."""
import unittest
try:
    import mujoco
except ImportError:
    mujoco = None


@unittest.skipIf(mujoco is None, 'MuJoCo 3.5 required for compiled geometry audit')
class StaticGeometryAuditTest(unittest.TestCase):
    def test_repeatable_both_registered_conditions_without_integration(self):
        from scripts.audit_g4s2a_gripper_v1_r1_geometry import evaluate
        original_step = mujoco.mj_step
        def forbidden_step(*args, **kwargs):
            raise AssertionError('physical integration is prohibited in this test')
        mujoco.mj_step = forbidden_step
        try:
            for case in ('I2', 'I3'):
                first, second = evaluate(case), evaluate(case)
                self.assertEqual(first, second)
                self.assertEqual(first['registered_finger_openings_m'], [.00499125, .00499125])
                self.assertAlmostEqual(first['peg_geom']['diameter_m'], .007986, places=14)
                self.assertEqual(first['compiled_timestep_s'], 1/120)
                for side in ('left', 'right'):
                    self.assertTrue(.0012 < first['sides'][side]['closest_distance_m'] < .0013)
                    self.assertEqual(len(first['sides'][side]['pad_geoms']), 5)
                    self.assertTrue(all(item['compiled_type'] == int(mujoco.mjtGeom.mjGEOM_BOX)
                                        for item in first['sides'][side]['pad_geoms']))
        finally:
            mujoco.mj_step = original_step

if __name__ == '__main__':
    unittest.main()
