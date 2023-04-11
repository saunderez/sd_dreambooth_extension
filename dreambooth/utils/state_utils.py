import os
import sys
sys.path.append('/workspaces/sd_dreambooth_extension','/workspaces/sd_dreambooth_extension/dreambooth','/workspaces/sd_dreambooth_extension/dreambooth/dataclasses', '/workspaces/sd_dreambooth_extension/dreambooth/utils')

class DreamboothState:
  import torch
  from dreambooth import shared
  from dreambooth.dataclasses import db_config as DreamboothConfig
  from accelerate import Accelerator
  from torch.optim import Optimizer
  from torch.optim.lr_scheduler import _LRScheduler as Scheduler


  def __init__(self, config: DreamboothConfig=shared.db_model_config, optimizer: Optimizer=None, scheduler: Scheduler=None, accelerator: Accelerator=None, verbose=False):
    self.model_config = config
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.accelerator = accelerator
    self.verbose = verbose
    self.config = config
    db_config = self.model_config.DreamboothConfig
    self.model_dir = db_config.model_dir
    self.model_name = db_config.model_name
    self.shared_diffusers_path = db_config.shared_diffusers_path
    self.model_path = db_config.model_path


    if self.get_model_params is not None:
        self.model = os.path.join(self.model_dir, self.model_name)
    else:
        self.model = None

    if self.accelerator is not None:
      self.device = self.accelerator.

      if self.optimizer is None:
        self.optimizer = self.accelerator.get_optimizer()

      if self.scheduler is None:
        self.scheduler = self.accelerator.get_scheduler()

    if self.verbose:
      print("DreamboothState initialized.")

  #def save_model_params(self):
  #  if self.model is None or self.model_dir is None:
  #    return
  #  model_dir = os.path.join(self.model_dir, 'model_params.pth')
  #  params = self.model.state()
  #  torch.save(params, model_dir)
  #
  #  if self.verbose:
  #    print("Model params saved: ", model_dir)

  def save_state(self):
    if self.optimizer is None or self.scheduler is None or self.model_dir is None:
      return

    optimizer_path = os.path.join(self.model_dir, 'optimizer.pth.tar')
    scheduler_path = os.path.join(self.model_dir, 'scheduler.pth.tar')

    optimizer_state_dict = self.optimizer.state_dict()
    self.save_state(optimizer_state_dict, optimizer_path)

    if self.verbose:
      print("Optimizer state saved: ", optimizer_path)

    scheduler_state = self.scheduler.state_dict()
    self.save_state(scheduler_state, scheduler_path)

    if self.verbose:
      print("Scheduler state saved: ", scheduler_path)

    self.save_model_params()

    if self.verbose:
      print(f"Checkpoint saved: optimizer={optimizer_path}, scheduler={scheduler_path} in model_dir={self.model_dir}")

  def save_state(self, state_dict, path):
    self.torch.save(state_dict, path)

    if self.verbose:
      print("State dictionary saved: ", path)

  def load_state_dict(self, path):
    return self.torch.load(path, map_location=self.device)

  def load_optimizer_state(self):
    if self.model_dir is None:
      return None

    optimizer_path = os.path.join(self.model_dir, 'optimizer.pth.tar')
    if os.path.exists(optimizer_path):
      state_dict = self.load_state(optimizer_path)
      self.optimizer.load_state(state_dict['optimizer_state_dict'])

      if self.verbose:
        print("Optimizer state dictionary loaded: ", optimizer_path)

    return self.optimizer

  def load_scheduler_state(self):
    if self.model_dir is None:
      return None

    scheduler_path = os.path.join(self.model_dir, 'scheduler.pth.tar')
    if os.path.exists(scheduler_path):
      state_dict = self.load_state(scheduler_path)
      self.scheduler.load_state(state_dict['scheduler_state'])

      if self.verbose:
        print("Scheduler state dictionary loaded: ", scheduler_path)

    return self.scheduler

  def load_model_params(self):
    if self.model is None or self.model_dir is None:
      return

    file_path = os.path.join(self.model_dir, 'model_params.pth')
    if os.path.exists(file_path):
      params = self.torch.load(file_path, map_location=self.device)
      self.model.load_state(params)

      if self.verbose:
        print("Model params loaded: ", file_path)

    return self.model

  def get_optimizer_params(self):
    if self.optimizer:
      return self.optimizer.state()
    else:
      return None

  def get_scheduler_params(self):
    if self.scheduler:
      return self.scheduler.state()
    else:
      return None

  def get_model_params(self):
    if self.model:
      return self.model.state()
    else:
      return None
