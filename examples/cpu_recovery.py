#!/usr/bin/env python3
"""Bounded CPU SGD recovery and PoL transition replay on synthetic inputs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from polbfl.protocol import RoundContext
from polbfl.verification import CheckpointOpening, IntervalWitness
from polbfl.verification.torch_replay import TorchSGDReplay, TorchSGDReplayConfig


def model_factory():
    return torch.nn.Linear(2,2,bias=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    target=args.output/'cpu-training.pt'
    if target.exists():raise SystemExit('Choose an output directory without an existing training checkpoint.')
    torch.set_num_threads(2);torch.manual_seed(1337)
    model=model_factory();optimizer=torch.optim.SGD(model.parameters(),lr=0.1,momentum=0.9,weight_decay=0.001)
    loss_fn=torch.nn.CrossEntropyLoss()
    initial=copy.deepcopy(model.state_dict());initial_optimizer=copy.deepcopy(optimizer.state_dict())
    batches=((torch.tensor([[1.,0.],[0.,1.]]),torch.tensor([0,1])),
             (torch.tensor([[1.,1.],[-1.,1.]]),torch.tensor([1,0])))
    def train(m,opt,batch):
        opt.zero_grad(set_to_none=True);loss=loss_fn(m(batch[0]),batch[1]);loss.backward();opt.step()
    train(model,optimizer,batches[0])
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':1},target)
    train(model,optimizer,batches[1]);expected=copy.deepcopy(model.state_dict())
    checkpoint=torch.load(target,map_location='cpu',weights_only=True)
    restored=model_factory();restored.load_state_dict(checkpoint['model'])
    resumed_optimizer=torch.optim.SGD(restored.parameters(),lr=0.1,momentum=0.9,weight_decay=0.001)
    resumed_optimizer.load_state_dict(checkpoint['optimizer']);train(restored,resumed_optimizer,batches[1])
    resumed_equal=all(torch.equal(expected[k],restored.state_dict()[k]) for k in expected)
    context=RoundContext(protocol_version='1',round_id='cpu-archive-check',client_id='client',model_id='linear',
        global_model_digest=hashlib.sha256(b'cpu-archive-initial').hexdigest(),optimizer='SGD',learning_rate=0.1,
        local_epochs=1,batch_size=2,checkpoint_interval=2)
    start=CheckpointOpening(0,type('Record',(),{'step':0})(),(),initial,[],[],(),{})
    end=CheckpointOpening(1,type('Record',(),{'step':2})(),(),expected,[],[],(),{})
    witness=IntervalWitness(0,batches,initial_optimizer,{})
    replay=TorchSGDReplay(TorchSGDReplayConfig(model_factory=model_factory,
        criterion_factory=torch.nn.CrossEntropyLoss,momentum=0.9,weight_decay=0.001))
    actual=replay(context,start,end,witness)
    replay_equal=all(torch.equal(expected[k],actual[k]) for k in expected)
    result={'device':'cpu','input':'synthetic','optimizer_steps':2,'checkpoint_step':1,
        'resumed_weights_equal_uninterrupted':resumed_equal,'replayed_weights_equal_training':replay_equal,
        'paper_experiment':False,'passed':resumed_equal and replay_equal}
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
