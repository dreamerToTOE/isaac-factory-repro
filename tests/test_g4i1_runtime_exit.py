"""CPU fault injection for Kit cleanup/exit propagation."""
import pytest
from types import SimpleNamespace
from scripts.g4i1_runtime import close_with_exit_status

@pytest.mark.parametrize("code", [0,1])
def test_kit_receives_exit_status_before_cleanup(code):
    calls=[]
    app=SimpleNamespace(app=SimpleNamespace(post_quit=lambda value:calls.append(("quit",value))), close=lambda:calls.append(("close",)))
    with pytest.raises(SystemExit) as exc: close_with_exit_status(app,code)
    assert exc.value.code==code
    assert calls==[("quit",code),("close",)]

def test_kit_zero_system_exit_cannot_mask_original_failure():
    def close(): raise SystemExit(0)
    app=SimpleNamespace(app=SimpleNamespace(post_quit=lambda value:None), close=close)
    with pytest.raises(SystemExit) as exc:close_with_exit_status(app,1)
    assert exc.value.code==1

def test_cleanup_exception_is_nonzero():
    def close(): raise RuntimeError("cleanup failed")
    app=SimpleNamespace(app=SimpleNamespace(post_quit=lambda value:None), close=close)
    with pytest.raises(SystemExit) as exc:close_with_exit_status(app,0)
    assert exc.value.code==1
