"""Do not let a nominal0 exit hide missing results or a saved exception."""
import json
from scripts.g4i1_acceptance import process_gate


def files(tmp_path):
    run=tmp_path/'run';sup=tmp_path/'supervision';run.mkdir();(sup/'supervisor').mkdir(parents=True);(sup/'guard').mkdir()
    def write(path,obj):path.write_text(json.dumps(obj))
    write(run/'result.json',{})
    write(sup/'supervisor/status.json',{'raw_wait_returncode':0,'termination_reason':'child_exit','error':None})
    write(sup/'supervisor/child_exit.json',{'raw_wait_returncode':0})
    write(sup/'guard/status.json',{'continuous_monitoring':True,'monitoring_gaps':[],'error':None})
    return run,sup


def test_saved_runtime_exception_rejects_false_zero_exit(tmp_path):
    run,sup=files(tmp_path);(run/'runtime_error.json').write_text('{"type":"RuntimeError"}')
    result=process_gate(run,sup);assert not result['passed'] and 'runtime_error artifact exists' in result['reasons']


def test_missing_completion_rejects_false_zero_exit(tmp_path):
    run,sup=files(tmp_path);(run/'result.json').unlink()
    assert not process_gate(run,sup)['passed']


def test_true_nonzero_child_wait_is_rejected(tmp_path):
    run,sup=files(tmp_path);(sup/'supervisor/child_exit.json').write_text('{"raw_wait_returncode":7}')
    assert not process_gate(run,sup)['passed']


def test_guard_failure_cannot_be_hidden_by_child_success(tmp_path):
    run,sup=files(tmp_path);(sup/'guard/status.json').write_text('{"continuous_monitoring":false,"monitoring_gaps":[1],"error":null}')
    assert not process_gate(run,sup)['passed']
