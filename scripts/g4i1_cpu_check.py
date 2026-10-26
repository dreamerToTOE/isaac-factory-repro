"""Actual frozen-checkpoint CPU parity against the evaluator's official Player.

No simulator, GPU, Runner, optimizer or training is used. Synthetic observations
verify actor arithmetic; simulator/controller coverage is reported separately.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys

import gym
import numpy as np
import torch
from rl_games.algos_torch.players import PpoPlayerContinuous
from scripts.g4i1_policy import FrozenPolicy
from scripts.g4i1_parity import compare_stages
from scripts.evaluation_runtime import reset_player_rnn_states


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Use new output path')
    torch.set_num_threads(2)
    repo=Path(__file__).resolve().parents[1]
    frozen=json.loads((repo/'evidence/g4_i/i0/checkpoint_audit.json').read_text())
    results={}
    for role in ['g2_epoch200_primary','g3c_r4_epoch400_secondary']:
        policy=FrozenPolicy.from_checkpoint(role=role,device='cpu')
        params=copy.deepcopy(frozen['checkpoints'][role]['agent_configuration']['params'])
        params['config'].update(device='cpu',device_name='cpu',num_actors=1,
            env_info={'observation_space':gym.spaces.Box(-np.inf,np.inf,(19,),np.float32),
                      'action_space':gym.spaces.Box(-1.,1.,(6,),np.float32),'agents':1,'value_size':1})
        oracle=PpoPlayerContinuous(params)
        oracle.model.load_state_dict(policy.model.state_dict(),strict=True);oracle.model.eval()
        oracle.get_batch_size(torch.zeros(1,19),1);oracle.init_rnn()
        torch.manual_seed(1001)
        state=policy.initial_state(1);ref={};actual={};rng_equal=[]
        for index in range(10):
            observation=torch.tensor(frozen['checkpoints'][role]['rms']['running_mean'],dtype=torch.float32)[None].clone()
            observation+=torch.linspace(-.1,.1,19)[None]*(index-4)
            observation[:,3:7]/=observation[:,3:7].norm(dim=-1,keepdim=True)
            capture={}
            hook=oracle.model.register_forward_hook(lambda m,i,o:capture.update(o))
            before=torch.get_rng_state().clone()
            with torch.random.fork_rng(devices=[]),torch.inference_mode():
                norm=oracle.model.norm_obs(observation.clone())
                action=oracle.get_action(observation.clone(),is_deterministic=True)
                expected_rng=torch.get_rng_state().clone()
            hook.remove();assert torch.equal(before,torch.get_rng_state())
            with torch.inference_mode():step=policy.infer(observation,state)
            rng_equal.append(torch.equal(expected_rng,torch.get_rng_state()))
            for name,a,b in [('observation',observation,step.observation),('normalized',norm,step.normalized),
                    ('mu',capture['mus'],step.mu),('h',capture['rnn_states'][0],step.state[0]),
                    ('c',capture['rnn_states'][1],step.state[1]),('clipped',action,step.clipped_action)]:
                ref.setdefault(name,[]).append(a.detach().cpu().numpy().copy())
                actual.setdefault(name,[]).append(b.detach().cpu().numpy().copy())
            state=step.state
            if index==4:
                state=policy.reset_state(state,torch.tensor([True]))
                reset_player_rnn_states(oracle,[True])
                assert not state[0].count_nonzero() and not state[1].count_nonzero()
        parity=compare_stages({k:np.stack(v) for k,v in ref.items()}, {k:np.stack(v) for k,v in actual.items()},
                              device_kind='cpu', require_all_stages=False,cases=['synthetic_observation','recurrent_reset'])
        results[role]={'identity':policy.identity,'parity':parity,'rng_comparisons':len(rng_equal),'rng_all_equal':all(rng_equal),
                       'rms_unchanged':policy.rms_unchanged(),'synthetic_observations':True,'actor_calls':10}
        assert all(rng_equal) and policy.rms_unchanged() and parity["passed"]
        del policy,oracle
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'scope':'CPU actual frozen actor vs original evaluator Player, no simulator/controller',
       'no_training_runner_module': 'rl_games.torch_runner' not in sys.modules,'results':results},indent=2)+'\n')
    print('CPU_CHECK_SAVED='+str(args.output))


if __name__=='__main__':main()
