import os
import torch
import DBConfig

class DBState:

  def __init__(self, model=None, optimizer=None, scheduler=None, accelerator=None, verbose=False, config=None):
    
    self.model = model
    self.optimizer = optimizer
    self.scheduler = scheduler
    
    self.accelerator = accelerator
    self.verbose = verbose
    self.device = shared.device
    
    if config is not None:
        self.model_dir = os.path.join(config.model_dir, config.model_name)
    else:
        self.model_dir = None
      
    if self.accelerator is not None:
      self.device = self.accelerator.device
      
      if self.optimizer is None:
        self.optimizer = self.accelerator.get_optimizer()
        
      if self.scheduler is None:
        self.scheduler = self.accelerator.get_scheduler()
    
    if self.verbose:
      print("DBState initialized.")

  def save_model_params(self):
    if self.model is None or self.model_dir is None:
      return

    model_dir = os.path.join(self.model_dir, 'model_params.pth')
    params = self.model.state_dict()
    torch.save(params, model_dir)

    if self.verbose:
      print("Model params saved: ", model_dir)

  def save_checkpoint(self):
    if self.optimizer is None or self.scheduler is None or self.model_dir is None:
      return
    
    optimizer_filename = os.path.join(self.model_dir, 'optimizer.pth.tar')
    scheduler_filename = os.path.join(self.model_dir, 'scheduler.pth.tar')

    optimizer_state_dict = self.optimizer.state_dict()
    self.save_state_dict(optimizer_state_dict, optimizer_filename)
    
    if self.verbose:
      print("Optimizer state saved: ", optimizer_filename)

    scheduler_state_dict = self.scheduler.state_dict()
    self.save_state_dict(scheduler_state_dict, scheduler_filename)

    if self.verbose:
      print("Scheduler state saved: ", scheduler_filename)

    self.save_model_params()
    
    if self.verbose:
      print(f"Checkpoint saved: optimizer={optimizer_filename}, scheduler={scheduler_filename} in model_dir={self.model_dir}")

  def save_state_dict(self, state_dict, filename):
    torch.save(state_dict, filename)

    if self.verbose:
      print("State dictionary saved: ", filename)

  def load_state_dict(self, filename):
    return torch.load(filename, map_location=self.device)

  def load_optimizer_state(self):
    if self.model_dir is None:
      return None
    
    optimizer_filename = os.path.join(self.model_dir, 'optimizer.pth.tar')
    if os.path.exists(optimizer_filename):
      state_dict = self.load_state_dict(optimizer_filename)
      self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])

      if self.verbose:
        print("Optimizer state dictionary loaded: ", optimizer_filename)
    
    return self.optimizer

  def load_scheduler_state(self):
    if self.model_dir is None:
      return None
    
    scheduler_filename = os.path.join(self.model_dir, 'scheduler.pth.tar')
    if os.path.exists(scheduler_filename):
      state_dict = self.load_state_dict(scheduler_filename)
      self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])

      if self.verbose:
        print("Scheduler state dictionary loaded: ", scheduler_filename)
    
    return self.scheduler

  def load_model_params(self):
    if self.model is None or self.model_dir is None:
      return

    file_path = os.path.join(self.model_dir, 'model_params.pth')
    if os.path.exists(file_path):
      params = torch.load(file_path, map_location=self.device)
      self.model.load_state_dict(params)

      if self.verbose:
        print("Model params loaded: ", file_path)

    return self.model

  def get_optimizer_params(self):
    if self.optimizer:
      return self.optimizer.state_dict()
    else:
      return None

  def get_scheduler_params(self):
    if self.scheduler:
      return self.scheduler.state_dict()
    else:
      return None

  def get_model_params(self):
    if self.model:
      return self.model.state_dict()
    else:
      return None
