import os
from abc import ABC
from logging import getLogger
from typing import Any, Dict, Union
from pathlib import Path
import torch
import torchmetrics
import wandb
from omegaconf import OmegaConf
from pytorch_lightning.lite import LightningLite

from src.solver.build import build_lr_scheduler, make_optimizer
from src.utils.utils import init_wandb, init_steps


class BaseLite(LightningLite, ABC):
    def __init__(self, cfg, model, dataloaders, step=0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        init_steps(cfg, dataloaders[0])
        self.step = step
        self.logger = getLogger(cfg.MODEL.NAME)
        if self.is_global_zero:
            total = sum([param.nelement() for param in model.parameters()])
            self.logger.info(model.config)
            self.logger.info("Number of parameter: %.2fM" % (total / 1e6))
        optimizer = make_optimizer(cfg, model)
        self.model = model
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        if self.is_global_zero:
            init_wandb(cfg)
        self.lr_scheduler = build_lr_scheduler(cfg, optimizer, offset=self.step)
        if self.step > 0:
            self.move_optimizer_state_dict()

    def init_after_ddp(self):
        self.model, self.optimizer = self.setup(self.model, self.optimizer)
        self.dataloaders = self.setup_dataloaders(*self.dataloaders, replace_sampler=False)


    def train(self):
        self.model.train()
        for i, batch in enumerate(self.dataloaders[0]):
            outputs = self.model(**batch['net_input'], labels=batch['target']['finetune_target'])

            self.backward(outputs.loss)
            if (i + 1) % self.cfg.SOLVER.AGB == 0 or (i + 1) == len(self.dataloaders[0]):
                # model._precision_plugin.clip_gradients(optimizer, 1)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                self.step += 1
            if ((self.step % 100) == 0) and (i % self.cfg.SOLVER.AGB) == 0:
                log_dict = {
                    'loss': outputs.loss.item(),
                    'lr': self.optimizer.state_dict()["param_groups"][0]["lr"],
                }
                wandb.log(log_dict, step=self.step)

    def save(self, content: Dict[str, Any] = None, filepath: Union[str, Path] = None, prefix:str=None):
        if self.is_global_zero:
            model_to_save = self.model.module
            # model_to_save = model_to_save.module
            hf_dir = filepath if filepath else self.cfg.OUTPUT_DIR
            if prefix is not None:
                hf_dir = os.path.join(hf_dir, prefix)
            if not os.path.exists(hf_dir):
                os.makedirs(hf_dir)
            model_to_save.save_pretrained(hf_dir)
            torch.save(self.optimizer.state_dict(), os.path.join(hf_dir, 'optimizer.pt'))
            OmegaConf.save(config=self.cfg, f=os.path.join(hf_dir, 'global_config.yml'))
            if content is not None:
                super(BaseLite, self).save(content, filepath)
            self.logger.info(f'model ckpt save to {hf_dir}')

    def move_optimizer_state_dict(self):
        """
        移动optim的参数至当前device
        """
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if k == 'step':
                    continue
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
