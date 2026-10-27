"""CPU-only closeout regression. Invoke this file, never global discovery.

No simulator, production evaluator, checkpoint deserializer, or policy import.
"""
import copy
import importlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import g4_sim2sim_closeout as audit


class CloseoutChecks(unittest.TestCase):
    def test_original_gates_and_source_denominator(self):
        s = audit.read('evidence/g4_s/closeout/milestone_summary.json')
        self.assertTrue(audit.validate_summary(s))
        self.assertEqual(s['original_source_baseline']['original_epoch200_fields']['aggregate']['episodes_completed'], 288)
        self.assertEqual(s['historical_gates']['GRIPPER_V1_R1']['original_fields']['preparations']['PREP_I3_RATE_LIMITED']['steps'], 23)

    def test_false_transfer_success_rejected(self):
        s = copy.deepcopy(audit.make_summary())
        s['final_status'] = {**s['final_status'], 'TASK_LEVEL_ZERO_SHOT_TRANSFER': 'SUCCESS'}
        with self.assertRaises(ValueError):
            audit.validate_summary(s)

    def test_rewritten_historical_gate_rejected(self):
        s = copy.deepcopy(audit.make_summary())
        s['historical_gates']['GRIPPER_V1']['original_fields']['i3_status'] = 'PASS'
        with self.assertRaises(ValueError):
            audit.validate_summary(s)

    def test_protected_file_and_checkpoint_changes_rejected(self):
        before = {'tracked': {'evidence/old.json': 'a', 'README.md': 'b'},
                  'external': {'checkpoint': {'sha256': 'c'}}}
        after = copy.deepcopy(before)
        after['tracked']['README.md'] = 'allowed'
        self.assertEqual(audit.preservation_diff(before, after), ['README.md'])
        after['tracked']['evidence/old.json'] = 'bad'
        with self.assertRaises(ValueError):
            audit.preservation_diff(before, after)
        after = copy.deepcopy(before)
        after['external']['checkpoint']['sha256'] = 'bad'
        with self.assertRaises(ValueError):
            audit.preservation_diff(before, after)

    def test_missing_local_link_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            doc = Path(d) / 'doc.md'
            self.assertEqual(audit.check_links('[missing](no.json)', doc), ['no.json'])
            (Path(d) / 'yes.json').write_text('{}')
            self.assertEqual(audit.check_links('[ok](yes.json) [web](https://example.com)', doc), [])

    def test_runtime_import_barrier(self):
        for name in ['mujoco', 'torch', 'isaacsim', 'isaaclab', 'rl_games']:
            with self.assertRaises(ImportError):
                importlib.import_module(name)
        self.assertFalse(set(sys.modules) & {'mujoco', 'torch', 'isaacsim', 'isaaclab', 'rl_games'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
