import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from pytorch_optimizer import Adan
from torch_optimizer import RAdam
from lion_pytorch import Lion
import logging

class Optimizers:
    """
    Class to initialize various optimizers for model training.
    """
    def __init__(self, optimizer_name, parameters, lr, weight_decay=0.0, **kwargs):
        self.logger = logging.getLogger('TrainLogger')
        self.optimizer_name = optimizer_name.lower()
        self.lr = lr
        self.weight_decay = weight_decay
        self.kwargs = kwargs
        self.parameters = parameters
        print(self.lr)
    def get_optimizer(self):
        """
        Returns the specified optimizer for the given parameters.
        """
       
        if self.optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                self.parameters,
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=self.kwargs.get('betas', (0.9, 0.999)),
                eps=self.kwargs.get('eps', 1e-8)
            )
        elif self.optimizer_name == 'radam':
            optimizer = RAdam(
                self.parameters,
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=self.kwargs.get('betas', (0.9, 0.999)),
                eps=self.kwargs.get('eps', 1e-8)
            )
        elif self.optimizer_name == 'lion':
            optimizer = Lion(
                self.parameters,
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=self.kwargs.get('betas', (0.9, 0.99))
            )
        elif self.optimizer_name == 'sparseadam':
            # Warning: SparseAdam only works with parameters that have sparse gradients
            try:
                optimizer = optim.SparseAdam(
                    self.parameters,
                    lr=self.lr,
                    betas=self.kwargs.get('betas', (0.9, 0.999)),
                    eps=self.kwargs.get('eps', 1e-8)
                )
            except ValueError as e:
                self.logger.error("SparseAdam requires sparse gradients. Falling back to Adam.")
                optimizer = optim.Adam(
                    self.parameters,
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    betas=self.kwargs.get('betas', (0.9, 0.999)),
                    eps=self.kwargs.get('eps', 1e-8)
                )
        elif self.optimizer_name == 'adan' and Adan is not None:
            optimizer = Adan(
                self.parameters,
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=self.kwargs.get('betas', (0.98, 0.92, 0.99)),
                eps=self.kwargs.get('eps', 1e-8)
            )
        else:
            self.logger.warning(f"Optimizer {self.optimizer_name} not available or not installed. Using Adam.")
            optimizer = optim.Adam(
                self.parameters,
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=self.kwargs.get('betas', (0.9, 0.999)),
                eps=self.kwargs.get('eps', 1e-8)
            )
        
        self.logger.info(f"Initialized {self.optimizer_name} optimizer with lr={self.lr}, weight_decay={self.weight_decay}")
        return optimizer

class LRSchedulers:
    """
    Class to initialize learning rate schedulers with optional warmup.
    """
    def __init__(self, scheduler_name, optimizer, epochs, warmup_epochs=0, warmup_start_factor=0.1, use_scheduler=True, **kwargs):
        self.logger = logging.getLogger('TrainLogger')
        self.scheduler_name = scheduler_name.lower()
        self.optimizer = optimizer
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.warmup_start_factor = warmup_start_factor
        self.use_scheduler = use_scheduler
        self.kwargs = kwargs

    def get_scheduler(self):
        """
        Returns the specified scheduler, optionally preceded by a warmup phase, or None if scheduler is disabled.
        """
        if not self.use_scheduler:
            self.logger.info("Learning rate scheduler disabled; using constant learning rate")
            return None

        main_scheduler = None
        if self.scheduler_name == 'cosineannealing':
            main_scheduler = lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs - self.warmup_epochs,
                eta_min=self.kwargs.get('eta_min', 0.0)
            )
        elif self.scheduler_name == 'reduceonplateau':
            main_scheduler = lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.kwargs.get('factor', 0.5),
                patience=self.kwargs.get('patience', 5),
            )
        elif self.scheduler_name == 'linear':
            main_scheduler = lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=self.kwargs.get('end_factor', 0.1),
                total_iters=self.epochs - self.warmup_epochs
            )
        else:
            self.logger.warning(f"Scheduler {self.scheduler_name} not recognized. Using CosineAnnealingLR.")
            main_scheduler = lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs - self.warmup_epochs,
                eta_min=self.kwargs.get('eta_min', 0.0)
            )

        if self.warmup_epochs > 0:
            warmup_scheduler = lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=self.warmup_start_factor,
                end_factor=1.0,
                total_iters=self.warmup_epochs
            )
            scheduler = lr_scheduler.ChainedScheduler([warmup_scheduler, main_scheduler])
            self.logger.info(f"Initialized {self.scheduler_name} scheduler with {self.warmup_epochs} warmup epochs "
                             f"(start_factor={self.warmup_start_factor})")
        else:
            scheduler = main_scheduler
            self.logger.info(f"Initialized {self.scheduler_name} scheduler without warmup")

        return scheduler