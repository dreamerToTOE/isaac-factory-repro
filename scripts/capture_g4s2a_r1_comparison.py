"""Actual desktop capture of SAVED diagnostic states; zero new integration.

This is explicitly not a screenshot contemporaneous with original integration.
Only this process's own native MuJoCo window is raised for capture.
"""
from pathlib import Path
import argparse,ctypes,hashlib,json,sys,threading,time
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.g4s2a_r1_viewer import require_gui_environment,load_archive,show,configure_camera,window_info,capture_desktop_window,close_and_join,dump,sha,XML_SHA
from scripts.g4s2a_r1_diagnostics import model_for_variant


def raise_own_window(wid):
    lib=ctypes.CDLL('libX11.so.6')
    lib.XOpenDisplay.argtypes=[ctypes.c_char_p];lib.XOpenDisplay.restype=ctypes.c_void_p
    lib.XRaiseWindow.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
    lib.XSetInputFocus.argtypes=[ctypes.c_void_p,ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong]
    lib.XSync.argtypes=[ctypes.c_void_p,ctypes.c_int];lib.XCloseDisplay.argtypes=[ctypes.c_void_p]
    display=lib.XOpenDisplay(None)
    if not display:raise RuntimeError('GUI_BLOCKED_XOPEN_DISPLAY')
    try:lib.XRaiseWindow(display,int(wid,16));lib.XSetInputFocus(display,int(wid,16),2,0);lib.XSync(display,False)
    finally:lib.XCloseDisplay(display)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    require_gui_environment();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    import mujoco.viewer
    matrix=ROOT/'evidence/g4_s/s2a_r1/diagnostics/gui_matrix_v1'
    prereg=json.loads((matrix/'preregistered_matrix.json').read_text())
    manifest=[];closes=[]
    for variant_id in ('D01','D06','ARCHIVED_S2A'):
        variant=next(v for v in prereg['variants'] if v['id']==('D01' if variant_id=='ARCHIVED_S2A' else variant_id))
        env,identity=model_for_variant(variant)
        if variant_id=='ARCHIVED_S2A':
            _,rows,_=load_archive();selected=[(8,rows[7]['state'])];config_id=XML_SHA;mode='ARCHIVED S2A STATE PLAYBACK; ZERO NEW INTEGRATION'
            input_path=ROOT/'evidence/g4_s/s2a/run_20261026T140338/seed_1001/physics.jsonl.gz'
        else:
            rows=[json.loads(x) for x in (matrix/variant_id/'physics.jsonl').read_text().splitlines() if json.loads(x)['event']=='physics_result']
            selected=[(k,rows[k-1]['state']) for k in (4,16)]
            original=json.loads((matrix/variant_id/'identity.json').read_text())
            if identity['model_configuration_sha256']!=original['model']['model_configuration_sha256']:raise ValueError('DIAGNOSTIC_PLAYBACK_MODEL_MISMATCH')
            config_id=identity['model_configuration_sha256'];mode='DIAGNOSTIC STATE PLAYBACK; ZERO NEW INTEGRATION'
            input_path=matrix/variant_id/'physics.jsonl'
        viewer=None;render_threads=[]
        try:
            before={t.ident for t in threading.enumerate()}
            viewer=mujoco.viewer.launch_passive(env.model,env.data,show_left_ui=False,show_right_ui=False)
            render_threads=[t for t in threading.enumerate() if t.ident not in before and '_launch_internal' in t.name]
            configure_camera(viewer,'closeup')
            for step,state in selected:
                show(viewer,env,state,label=variant_id+' recorded diagnostic state' if variant_id!='ARCHIVED_S2A' else 'historical physics8',step=step,mode=mode,
                     extra='Same saved qpos/qvel; historical solver impulses NOT reconstructed')
                time.sleep(.6);info=window_info();raise_own_window(info['window_id']);viewer.sync();time.sleep(.3)
                path=args.output/f'{variant_id}_physics_{step:02d}.png';capture_desktop_window(path,window_info())
                manifest.append({'file':str(path.relative_to(ROOT)),'sha256':sha(path),'source_experiment':variant_id,
                    'mode':'ARCHIVED_DIAGNOSTIC_STATE_PLAYBACK' if variant_id!='ARCHIVED_S2A' else 'ARCHIVED_S2A_STATE_PLAYBACK',
                    'physics_step':step,'episode_simulation_time_s':step/120,'recorded_absolute_time_s':state['time_s'],
                    'model_configuration_sha256':config_id,'source_jsonl':str(input_path.relative_to(ROOT)),'source_jsonl_sha256':sha(input_path),
                    'new_physics_steps':0,'policy_calls':0,'screenshot_source':'Actual Ubuntu X11 pixels of own raised native MuJoCo window',
                    'solver_internal_state':'UNKNOWN_NOT_RESTORED','window':info})
        finally:
            if viewer is not None:closes.append(close_and_join(viewer,render_threads))
    dump(args.output/'capture_manifest.json',manifest)
    dump(args.output/'process_receipt.json',{'exit_code':0,'actual_visible_GUI':True,'new_physics_steps':0,'policy_calls':0,'source_sha256':sha(Path(__file__)),'graceful_closes':closes})
    print(json.dumps({'visible_GUI':True,'capture_count':len(manifest),'new_physics_steps':0,'actor_calls':0}),flush=True)

if __name__=='__main__':main()
