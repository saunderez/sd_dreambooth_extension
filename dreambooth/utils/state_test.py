import sys
sys.path.append('/workspaces/sd_dreambooth_extension','/workspaces/sd_dreambooth_extension/dreambooth','/workspaces/sd_dreambooth_extension/dreambooth/dataclasses', '/workspaces/sd_dreambooth_extension/dreambooth/utils')


import os
import torch
import torch.nn
from accelerate import Accelerator

from dreambooth.shared import shared
from state_utils import DreamboothState as dreambooth_state

model_name = 'TestModel123'
model_dir = '/workspaces/model'

model_path = os.path.join(model_dir, model_name)

model = torch.nn.Linear(in_features=10,out_features=10,bias=True,device=shared.device,dtype=torch.float16)
device = shared.device
optimizer = torch.optim.SGD(model, lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
accelerator = Accelerator(optimizer=optimizer, scheduler=scheduler, model=model, device=device)


dreambooth_state = dreambooth_state(accelerator=accelerator, verbose=True)


def test_load_accelerator(optimizer, scheduler, model_path):
    model_path = model_path
    dreambooth_state = dreambooth_state()
    optimizer_path = os.path.join('model_path', 'optimizer.pth.tar')
    scheduler_path = os.path.join('model_path', 'scheduler.pth.tar')
    assert os.path.exists(optimizer_path)
    assert os.path.exists(scheduler_path)

    if os.path.exists(optimizer_path) and os.path.exists(scheduler_path):
        dreambooth_state.load_state(optimizer, optimizer_path, scheduler, scheduler_path)
        assert dreambooth_state.optimizer == optimizer
        assert dreambooth_state.scheduler == scheduler
    else:
        print('Optimizer and scheduler state files not found. Skipping test.')



def test_save_accelerator(accelerator, model_path):
    accelerator = accelerator
    model_path = model_path
    dreambooth_state = dreambooth_state(accelerator=accelerator, verbose=True)
    optimizer = dreambooth_state.optimizer
    scheduler = dreambooth_state.scheduler
    if optimizer is not None and scheduler is not None:
        if os.path.exists(model_path):
            optimizer_path = os.path.join('model_path', 'optimizer.pth.tar')
            scheduler_path = os.path.join('model_path', 'scheduler.pth.tar')
            dreambooth_state.save_state(accelerator, optimizer_path, scheduler_path)
            if os.path.exists(optimizer_path) and os.path.exists(scheduler_path):
                print('Optimizer and scheduler state files saved.')
            else:
                print('Optimizer and scheduler state files not found. Skipping test.')
        else:
            print('Model path not found. Skipping test.')
    else:
        print('Optimizer and scheduler state files not found. Skipping test.')



#Saves the optimizer state and scheduler state to file for the given model
def test_save_state(accelerator, model_path):
    accelerator = accelerator
    model_path = model_path
    optimizer = accelerator.get_optimizer()
    scheduler = accelerator.get_scheduler()
    dreambooth_state = dreambooth_state()
    if optimizer is not None and scheduler is not None:
        if os.path.exists(model_path):
            #create a path to save the optimizer and scheduler staets
            optimizer_path = os.path.join('model_path', 'optimizer.pth.tar')
            scheduler_path = os.path.join('model_path', 'scheduler.pth.tar')

            #save the fake optimizer state
            dreambooth_state.save_state(optimizer, optimizer_path)
            dreambooth_state.save_state(scheduler, scheduler_path)

            if dreambooth_state.optimizer == optimizer and dreambooth_state.scheduler == scheduler:
                print('Optimizer and Scheduler state saved.')
            else:
                if dreambooth_state.optimizer != optimizer:
                    print('Optimizer state not saved.')
                if dreambooth_state.scheduler != scheduler:
                        print('Scheduler state not saved.')
        else:
            print('Model path not found. Skipping test.')
    else:
        print('Optimizer and scheduler state files not found. Skipping test.')

def test_load_state(accelerator, model_path, model_config):
    model_path = model_path
    model_config = model_config
    accelerator = accelerator
    #create a path to save the optimizer and scheduler staets
    optimizer = accelerator.get_optimizer()
    scheduler = accelerator.get_scheduler()
    dreambooth_state = dreambooth_state(optimizer, scheduler)
    if optimizer is not None and scheduler is not None:
        optimizer_path = os.path.join('model_path', 'optimizer.pth.tar')
        scheduler_path = os.path.join('model_path', 'scheduler.pth.tar')
        if os.path.exists(optimizer_path) or os.path.exists(scheduler_path):
            dreambooth_state = dreambooth_state(optimizer, scheduler)

            #load the fake optimizer state
            dreambooth_state.load_state(optimizer.load_state_dict(torch.load(optimizer_path)))
            dreambooth_state.load_state(scheduler.load_state_dict(torch.load(scheduler_path)))

            if dreambooth_state.optimizer != optimizer and dreambooth_state.scheduler != scheduler:
                print('Optimizer and Scheduler states have changed after loading.')
            else:
                if dreambooth_state.optimizer == optimizer:
                    print('Optimizer state not loaded.')
                elif dreambooth_state.scheduler == scheduler:
                    print('Scheduler state not loaded.')
        else:
            print("Optimizer and scheduler state files not found. Skipping test.")
    else:
        if optimizer == None:
            print("Accelerator did not contain both optimizer.")
        elif scheduler == None:
            print("Accelerator did not contain scheduler.")
