"""CPU-only diagnostic geometry/action traces for completed I2 failures.

The original pre-reset official success bool is primary evidence. NumPy/Torch/
Isaac are not imported. Recomputed XY/Z conditions are a separate descriptive
check, never a replacement predicate or proof of a contact/jamming mechanism.
Raw batch traces stay external; only failed single-env step CSVs are extracted.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def threshold_provenance(identity_path=None):
    """Verify pinned upstream sources and read constants without importing Isaac."""
    path = Path(identity_path) if identity_path else REPO/'evidence/g4_i/i0/checkpoint_and_source_identity.json'
    identity = json.loads(path.read_text())
    sources = {}
    for label in ('factory_env.py','factory_tasks_cfg.py'):
        item = next(row for row in identity['source_inventory'] if row['label'] == label)
        source_path = Path(item['path'])
        content = source_path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != item['sha256']:
            raise ValueError(f'frozen diagnostic source hash mismatch: {label}')
        sources[label] = dict(path=str(source_path.resolve()),sha256=digest,ast=ast.parse(content))
    tree=sources['factory_tasks_cfg.py']['ast']
    def class_literal(class_name,name):
        klass=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name==class_name)
        for node in klass.body:
            if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id==name for target in node.targets):
                return float(ast.literal_eval(node.value))
            if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name) and node.target.id==name:
                return float(ast.literal_eval(node.value))
        raise ValueError(f'missing frozen literal {class_name}.{name}')
    height=class_literal('Hole8mm','height')
    fraction=class_literal('PegInsert','success_threshold')
    predicate=next(node for node in ast.walk(sources['factory_env.py']['ast']) if isinstance(node,ast.FunctionDef) and node.name=='_get_curr_successes')
    centered=next(node for node in ast.walk(predicate) if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='is_centered' for target in node.targets))
    comparisons=[node for node in ast.walk(centered.value) if isinstance(node,ast.Compare) and isinstance(node.left,ast.Name) and node.left.id=='xy_dist']
    if len(comparisons)!=1 or not isinstance(comparisons[0].ops[0],ast.Lt):
        raise ValueError('frozen XY comparison structure changed')
    xy=float(ast.literal_eval(comparisons[0].comparators[0]))
    if height != .025 or fraction != .04 or xy != .0025:
        raise ValueError('frozen PegInsert diagnostic constants changed')
    return dict(xy_threshold_m=xy,fixed_hole_height_m=height,held_peg_height_m=class_literal('Peg8mm','height'),success_threshold_fraction=fraction,
                z_threshold_m=height*fraction,comparison='strict xy < threshold and signed z < threshold; no absolute-z or lower-bound substitution',
                height_disambiguation='fixed Hole8mm height is0.025m; held Peg8mm height is0.05m and is not the predicate height',
                identity_path=str(path.resolve()),identity_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                sources={key:{k:v for k,v in value.items() if k!='ast'} for key,value in sources.items()},
                interpretation='independent diagnostic recomputation; does not replace or modify original official predicate; PegInsert check_rot=False')


def _vector(value,env_id,width,label):
    if not isinstance(value,list) or env_id>=len(value) or not isinstance(value[env_id],list) or len(value[env_id])!=width:
        raise ValueError(f'{label} missing selected env or incorrect width')
    result=value[env_id]
    if any(type(x) not in (float,int) or not math.isfinite(x) for x in result):
        raise ValueError(f'{label} contains non-finite or non-numeric values')
    return result


def _read_episodes(path):
    with Path(path).open() as stream: value=json.load(stream)
    return value if isinstance(value,list) else value['episodes']


def _episode_specs(episodes):
    specs=[]
    identities=set()
    for episode in episodes:
        if type(episode['terminal_success']) is not bool or type(episode['ever_success']) is not bool:
            raise ValueError('success flags must be booleans')
        if episode['terminal_success']:
            if not episode['ever_success']: raise ValueError('terminal success requires ever-success')
            continue
        env_id,episode_id=episode['env_id'],episode['episode_id']
        first,last=episode['trace']['first_policy_step'],episode['trace']['last_policy_step']
        if any(type(value) is not int or value<0 for value in (env_id,episode_id)):
            raise ValueError('env/episode identifiers must be nonnegative integers')
        if any(type(value) is not int or value<1 for value in (first,last)) or last<first or episode['steps']!=last-first+1:
            raise ValueError('failure episode trace range disagrees with its step count')
        identity=(episode.get('evaluation_seed',episode.get('seed','UNKNOWN')),env_id,episode_id)
        if identity in identities: raise ValueError('duplicate failure episode identity')
        identities.add(identity)
        specs.append(dict(episode=episode,env_id=env_id,first=first,last=last,states={},actions={},
                          start_byte=None,end_byte=None,start_line=None,end_line=None))
    return specs


def _scan_seed(run_dir,specs,thresholds):
    trace=Path(run_dir)/'i2_trace.jsonl'
    digest=hashlib.sha256()
    with trace.open('rb') as stream:
        line_number=0
        while True:
            start=stream.tell();line=stream.readline()
            if not line: break
            digest.update(line);line_number+=1
            if not line.strip(): continue
            try: row=json.loads(line)
            except (ValueError,UnicodeDecodeError) as exc:
                raise ValueError(f'invalid raw JSON line {line_number}') from exc
            if row.get('kind') not in ('action','success_state'): continue
            step=row['policy_step']
            if type(step) is not int or step<1: raise ValueError('invalid raw policy step')
            for spec in specs:
                if not spec['first']<=step<=spec['last']: continue
                if spec['start_byte'] is None: spec['start_byte'],spec['start_line']=start,line_number
                spec['end_byte'],spec['end_line']=stream.tell(),line_number
                env_id=spec['env_id']
                if row['kind']=='action':
                    if step in spec['actions']: raise ValueError('duplicate failure action step')
                    spec['actions'][step]={key:_vector(row[key],env_id,6,key) for key in ('raw_mu','clipped','ema')}
                else:
                    if step in spec['states']: raise ValueError('duplicate failure success-state step')
                    cap=row['capture'];geometry=cap['geometric_snapshots']
                    held=_vector(geometry['held_base_pos'],env_id,3,'held_base_pos')
                    target=_vector(geometry['target_held_base_pos'],env_id,3,'target_held_base_pos')
                    success=cap['success'][env_id]
                    if type(success) is not bool: raise ValueError('official success must be boolean')
                    number=cap['step_numbers'][env_id]
                    if type(number) is not int or number!=step-spec['first']+1:
                        raise ValueError('failure episode step counter disagrees with trace range')
                    xy=math.hypot(held[0]-target[0],held[1]-target[1]);z=held[2]-target[2]
                    centered=xy<thresholds['xy_threshold_m'];below=z<thresholds['z_threshold_m']
                    spec['states'][step]=dict(policy_step=step,episode_step=number,official_success=success,
                        held_base_x_m=held[0],held_base_y_m=held[1],held_base_z_m=held[2],
                        target_held_base_x_m=target[0],target_held_base_y_m=target[1],target_held_base_z_m=target[2],
                        xy_distance_m=xy,signed_z_displacement_m=z,diagnostic_xy_satisfied=centered,
                        diagnostic_z_satisfied=below,diagnostic_success=centered and below,
                        diagnostic_matches_official=(centered and below)==success)
    for spec in specs:
        expected=set(range(spec['first'],spec['last']+1))
        if set(spec['states'])!=expected or set(spec['actions'])!=expected:
            raise ValueError(f"incomplete failure evidence env={spec['env_id']} episode={spec['episode']['episode_id']}; missing states={sorted(expected-set(spec['states']))}; missing actions={sorted(expected-set(spec['actions']))}")
        ordered=[spec['states'][step] for step in sorted(expected)]
        flags=[row['official_success'] for row in ordered]
        if any(flags)!=spec['episode']['ever_success'] or flags[-1]!=spec['episode']['terminal_success']:
            raise ValueError('raw official predicate history disagrees with completed failure episode')
        first_success=next((i+1 for i,value in enumerate(flags) if value),None)
        if spec['episode'].get('first_success_step',first_success)!=first_success:
            raise ValueError('raw first success step disagrees with completed episode')
        spec['source']=dict(path=str(trace.resolve()),sha256=digest.hexdigest(),
            byte_range_start_inclusive=spec['start_byte'],byte_range_end_exclusive=spec['end_byte'],
            line_range_start_inclusive=spec['start_line'],line_range_end_inclusive=spec['end_line'],
            env_id=spec['env_id'],first_policy_step=spec['first'],last_policy_step=spec['last'],
            evidence='original external complete batch trace; extracted CSV contains only selected failure env geometry and action rows')
    return digest.hexdigest()


def analyze_runs(run_dirs,output_dir,*,identity_path=None):
    """Extract each failure's complete step series, scanning each raw seed once."""
    output_dir=Path(output_dir)
    if output_dir.exists(): raise FileExistsError(output_dir)
    thresholds=threshold_provenance(identity_path)
    failures=[];sources=[];identities=set();episode_total=0
    for run_dir in run_dirs:
        run_dir=Path(run_dir);episode_path=run_dir/'episodes.json'
        episodes=_read_episodes(episode_path);episode_total+=len(episodes)
        specs=_episode_specs(episodes)
        trace_sha=_scan_seed(run_dir,specs,thresholds)
        sources.append(dict(run_dir=str(run_dir.resolve()),trace_sha256=trace_sha,
                            episode_sha256=hashlib.sha256(episode_path.read_bytes()).hexdigest(),
                            completed_episodes=len(episodes),failed_episodes=len(specs)))
        for spec in specs:
            episode=spec['episode'];seed=episode.get('evaluation_seed',episode.get('seed','UNKNOWN'))
            identity=(seed,spec['env_id'],episode['episode_id'])
            if identity in identities: raise ValueError('duplicate failure identity across runs')
            identities.add(identity)
            steps=[]
            for step in range(spec['first'],spec['last']+1):
                record=dict(spec['states'][step])
                for key,values in spec['actions'][step].items():
                    record.update({f'{key}_{index}':value for index,value in enumerate(values)})
                steps.append(record)
            final=steps[-1]
            unmet=[]
            if not final['diagnostic_xy_satisfied']: unmet.append('XY_DISTANCE_NOT_BELOW_0_0025_M')
            if not final['diagnostic_z_satisfied']: unmet.append('SIGNED_Z_NOT_BELOW_0_001_M')
            geometric_class=('XY_AND_Z_CRITERIA_UNMET' if len(unmet)==2 else ('XY_CRITERION_UNMET' if unmet and unmet[0].startswith('XY') else 'Z_CRITERION_UNMET' if unmet else 'NO_DIAGNOSTIC_GEOMETRIC_CRITERION_UNMET'))
            failures.append(dict(evaluation_seed=seed,env_id=spec['env_id'],episode_id=episode['episode_id'],
                steps=len(steps),ever_success=episode['ever_success'],terminal_success=episode['terminal_success'],
                temporal_failure_class='lost_success_before_terminal' if episode['ever_success'] else 'never_success',
                terminal_geometric_class=geometric_class,terminal_unmet_criteria=unmet,
                terminal_xy_distance_m=final['xy_distance_m'],terminal_signed_z_displacement_m=final['signed_z_displacement_m'],
                minimum_xy_distance_m=min(row['xy_distance_m'] for row in steps),
                minimum_signed_z_displacement_m=min(row['signed_z_displacement_m'] for row in steps),
                maximum_signed_z_displacement_m=max(row['signed_z_displacement_m'] for row in steps),
                diagnostic_predicate_mismatches=sum(not row['diagnostic_matches_official'] for row in steps),
                first_success_step=episode.get('first_success_step'),
                pre_reset_terminal_state=episode.get('pre_reset_terminal_state','UNKNOWN'),
                terminal_recomputed_state=final,raw_trace=spec['source'],
                initial_randomized_configuration=episode.get('initial_randomized_configuration','UNKNOWN'),
                csv_name=f'seed{seed}_env{spec["env_id"]}_episode{episode["episode_id"]}_failure_steps.csv',
                interpretation='observed terminal geometric criteria and predicate history only; no contact, jam, force, friction, damage, or causal mechanism inferred',
                step_rows=steps))
    output_dir.mkdir(parents=True,exist_ok=False)
    for failure in failures:
        steps=failure.pop('step_rows');path=output_dir/failure['csv_name']
        with path.open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(steps[0]));writer.writeheader();writer.writerows(steps)
        failure['csv_path']=str(path.resolve());failure['csv_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    result=dict(schema='g4i2_failure_geometry_diagnostics_v1',completed_episodes=episode_total,
                failed_episodes=len(failures),failures=failures,sources=sources,thresholds=thresholds,
                raw_trace_policy='external originals retained and SHA256 referenced; no full batch trace is copied into this output',
                interpretation='diagnostic recomputation never replaces the captured official predicate; strict signed inequalities preserve frozen definitions; no causal contact/jam claims')
    with (output_dir/'failure_analysis.json').open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--identity',type=Path,help='frozen I0 identity; default repository identity')
    args=parser.parse_args()
    result=analyze_runs(args.run_dir,args.output,identity_path=args.identity)
    print(json.dumps(dict(completed_episodes=result['completed_episodes'],failed_episodes=result['failed_episodes'],output=str(args.output))))


if __name__=='__main__': main()
