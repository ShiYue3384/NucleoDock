import os
import gc
import re
import csv
import datetime
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.data import get_dataloader
from src.data import Dictionary
from src.modeling.modeling_foldock2_all import FoldForDocking
from src.modeling.modeling_hf_unimol import UnimolConfig
from src.solver.build import make_optimizer, build_lr_scheduler
from src.solver.ema import EMA
from src.utils.docking_utils import docking
from src.utils.utils import args_parse, get_abs_path, init_steps


class DockingModelWrapper(pl.LightningModule):
    def __init__(self, training_config, model_config):
        super().__init__()
        self.training_config = training_config
        self.model_config = model_config
        self.test_step_outputs = []
        self.model = FoldForDocking(model_config)

    @staticmethod
    def get_log_data(batch_outputs, prefix='train'):
        log_dict = {}
        if isinstance(batch_outputs, dict):
            for k, v in batch_outputs.items():
                if ('loss' in k) and (v is not None) and torch.is_tensor(v):
                    log_dict[f'{prefix}/{k}'] = v.item()
        else:
            for attr in ['loss', 'score_loss', 'cross_loss', 'holo_loss']:
                if hasattr(batch_outputs, attr):
                    val = getattr(batch_outputs, attr)
                    if val is not None and torch.is_tensor(val):
                        log_dict[f'{prefix}/{attr}'] = val.item()
        return log_dict

    def on_train_epoch_start(self):
        if hasattr(self.trainer.train_dataloader.dataset, 'set_epoch'):
            self.trainer.train_dataloader.dataset.set_epoch(self.current_epoch)

    def on_train_batch_end(self, outputs, batch, batch_idx: int, unused: int = 0) -> None:
        del outputs, batch
        if batch_idx % 200 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def training_step(self, batch, batch_idx):
        outputs = self._forward_pass(batch)
        loss = outputs.loss

        log_dict = self.get_log_data(outputs, prefix='train')
        
        opt = self.optimizers(use_pl_optimizer=False)
        if opt:
            lr = opt.state_dict()["param_groups"][0]["lr"]
            log_dict['train/lr'] = lr
            
        self.log_dict(log_dict, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self._forward_pass(batch)
        log_dict = self.get_log_data(outputs, prefix='valid')
        self.log_dict(log_dict, on_epoch=True, sync_dist=True, batch_size=self.training_config.SOLVER.VALID_BSZ)
        return outputs

    def test_step(self, batch, batch_idx):
        outputs = self._forward_pass(batch)

        content = dict()
        content["smi_name"] = batch["smi_name"]
        content["pocket_name"] = batch["pocket_name"]
        content['pi'] = outputs.mdn[0].detach().cpu()
        content['mu'] = outputs.mdn[1].detach().cpu()
        content['sigma'] = outputs.mdn[2].detach().cpu()
        content["cross_distance_predict"] = outputs.cross_distance_predict.data.float().detach().cpu()
        content["holo_distance_predict"]  = outputs.holo_distance_predict.data.float().detach().cpu()
        
        content["atoms"] = batch["net_input"]["mol_src_tokens"].data.detach().cpu()
        content["pocket_atoms"] = batch["net_input"]["pocket_src_tokens"].data.detach().cpu()
        
        content["holo_center_coordinates"] = batch["holo_center_coordinates"].data.float().detach().cpu()
        content["holo_init_coordinates"] = batch["net_input"]["holo_init_coord"].data.float().detach().cpu().numpy()
        content["holo_coordinates"] = batch["target"]["holo_coord"].data.float().detach().cpu()
        content["pocket_coordinates"] = batch["net_input"]["pocket_src_coord"].data.float().detach().cpu()
        
        self.test_step_outputs.append(content)
        return content


    def _forward_pass(self, batch):
        """
        统一处理输入参数的拆解，将 RNA Graph 和 Map 传递给模型
        """
        net_input = batch['net_input']
        target = batch['target']
        

        rna_batch_data = net_input.get('rna_batch_data', None)
        atom_to_res_map = net_input.get('atom_to_res_map', None)

        outputs = self.model(
            mol_src_tokens=net_input['mol_src_tokens'],
            mol_src_distance=net_input['mol_src_distance'],
            mol_src_edge_type=net_input['mol_src_edge_type'],
            pocket_src_tokens=net_input['pocket_src_tokens'],
            pocket_src_distance=net_input['pocket_src_distance'],
            pocket_src_edge_type=net_input['pocket_src_edge_type'],
            
            distance_target=target['distance_target'],
            holo_distance_target=target['holo_distance_target'],
            dist_threshold=self.training_config.MODEL.THRESHOLD,
            hop_matrix=target['hop_matrix'],
            
            rna_batch_data=rna_batch_data,
            atom_to_res_map=atom_to_res_map
        )
        return outputs

    def on_test_epoch_end(self) -> None:
        docking(get_abs_path(self.training_config.data.path),
                self.test_step_outputs,
                self.training_config.data.nthreads)
        self.test_step_outputs.clear()

    def on_train_end(self) -> None:
        if self.trainer.is_global_zero:
            OmegaConf.save(config=self.training_config,
                           f=os.path.join(self.training_config.OUTPUT_DIR, 'global_config.yml'))
            self.model_config.save_pretrained(self.training_config.OUTPUT_DIR)

    def configure_optimizers(self):
        optimizer = make_optimizer(self.training_config, self)
        lr_scheduler = build_lr_scheduler(self.training_config, optimizer, offset=0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "name": "WarmupLinearLR"
            }
        }


