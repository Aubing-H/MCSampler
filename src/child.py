import cv2
import os
import time
import gym
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import argparse
import multiprocessing as mp
import hydra
import pickle
import random
import sys
from copy import deepcopy
from functools import partial
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import torch.distributed as dist
from datetime import datetime
from hydra.utils import get_original_cwd, to_absolute_path
from pathlib import Path
from rich import print
from tqdm import tqdm
from functools import partial
from ray.rllib.models.torch.torch_action_dist import TorchMultiCategorical
from minedojo.minedojo_wrapper import MineDojoEnv

from src.models.simple import SimpleNetwork
from src.utils import negtive_sample, EvalMetric
from src.utils.vision import create_backbone, resize_image
from src.utils.loss import get_loss_fn
from src.data.data_lmdb import LMDBTrajectoryDataset
from src.data.dataloader import DatasetLoader
from src.eval.parallel_eval import ParallelEval

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

def making_exp_name(cfg):
    component = []
    if cfg['model']['use_horizon']:  # True
        component.append('p:ho')
    else:
        component.append('p:bc')
    
    component.append("b:" + cfg['model']['backbone_name'][:4])  # goal
    
    today = datetime.now()
    
    component.append(f"{today.month}-{today.day}#{today.hour}-{today.minute}")
    
    return "@".join(component)

from mineclip.mineclip.mineclip import MineCLIP
def accquire_goal_embeddings(clip_path, goal_list, device="cuda"):
    clip_cfg = {'arch': 'vit_base_p16_fz.v2.t2', 'hidden_dim': 512, 'image_feature_dim': 512, 'mlp_adapter_spec': 'v0-2.t0', 
               'pool_type': 'attn.d2.nh8.glusw', 'resolution': [160, 256]}
    clip_model = MineCLIP(**clip_cfg)
    clip_model.load_ckpt(clip_path, strict=True)
    clip_model = clip_model.to(device)
    res = {}
    with torch.no_grad():
        for goal in goal_list:
            res[goal] = clip_model.encode_text([goal]).cpu().numpy()
    return res

class Trainer:
    
    def __init__(self, cfg, device, local_rank=0):
        
        self.action_space = [3, 3, 4, 11, 11, 8, 1, 1]
        self.cfg = cfg
        self.device = device
        self.local_rank = local_rank  # 0
        self.exp_name = making_exp_name(cfg)

        #! accquire goal embeddings
        print("[Progress] [red]Computing goal embeddings using MineClip's text encoder...")
        # log, sheep, cow, pig; use mineclip to convert goal to embeddings
        self.embedding_dict = accquire_goal_embeddings(cfg['pretrains']['clip_path'], cfg['data']['filters'])
        
        if not cfg["eval"]["only"]:  # Train model
            #! use lmdb type dataset
            print("[Progress] [blue]Loading dataset...")
            # self.train_dataset = LMDBTrajectoryDataset(
            #     in_dir=cfg['data']['train_data'],
            #     # 1e4 * 32
            #     aug_ratio=cfg['optimize']['aug_ratio'] * cfg['optimize']['batch_size'],
            #     embedding_dict= self.embedding_dict,
            #     per_data_filters=cfg['data']['per_data_filters'],  # null
            #     skip_frame=cfg['data']['skip_frame'],  # 5
            #     window_len=cfg['data']['window_len'],  # 16
            #     padding_pos=cfg['data']['padding_pos'],  # left
            # )
            self.train_dataset = DatasetLoader(
                in_dir=cfg['data']['train_data'],
                # 1e4 * 32
                aug_ratio=cfg['optimize']['aug_ratio'] * cfg['optimize']['batch_size'],
                embedding_dict= self.embedding_dict,
                per_data_filters=cfg['data']['per_data_filters'],  # null
                skip_frame=cfg['data']['skip_frame'],  # 5
                window_len=cfg['data']['window_len'],  # 16
                padding_pos=cfg['data']['padding_pos'],  # left
            )
        
            if self.cfg.optimize.parallel:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
            else:
                self.train_sampler = torch.utils.data.sampler.RandomSampler(self.train_dataset)
            
            self.train_loader = DataLoader(
                self.train_dataset, 
                sampler=self.train_sampler, 
                pin_memory=True, 
                batch_size=cfg['optimize']['batch_size'], 
                num_workers=cfg['optimize']['num_workers'], 
            )
        
        backbone = create_backbone(
            name=cfg['model']['backbone_name'], 
            model_path=cfg['model']['model_path'], 
            weight_path=cfg['model']['weight_path'],
            goal_dim=cfg['model']['embed_dim'],
        )
        
        if cfg['model']['name'] == 'simple':
            # torch.compile()
            self.model = SimpleNetwork(
                action_space=self.action_space,
                state_dim=cfg['model']['state_dim'],
                goal_dim=cfg['model']['goal_dim'],
                action_dim=cfg['model']['action_dim'],
                num_cat=len(cfg['data']['filters']),
                hidden_size=cfg['model']['embed_dim'],
                fusion_type=cfg['model']['fusion_type'],
                max_ep_len=cfg['model']['max_ep_len'],
                backbone=backbone,
                frozen_cnn=cfg['model']['frozen_cnn'],
                use_recurrent=cfg['model']['use_recurrent'],
                use_extra_obs=cfg['model']['use_extra_obs'],
                use_horizon=cfg['model']['use_horizon'],
                use_prev_action=cfg['model']['use_prev_action'],
                extra_obs_cfg=cfg['model']['extra_obs_cfg'],
                use_pred_horizon=cfg['model']['use_pred_horizon'],
                c=cfg['model']['c'],
                transformer_cfg=cfg['model']['transformer_cfg']
            )
            # torch.save(self.model, "save_model.pt")
        else:
            raise NotImplementedError
        
        self.iter_num = -1
        
        if cfg['model']['load_ckpt_path'] != "":
            state_dict = torch.load(cfg['model']['load_ckpt_path'])
            print(f"[MAIN] load checkpoint from {cfg['model']['load_ckpt_path']}. ")
            print(f"[MAIN] iter_num: {state_dict['iter_num']}, loss: {state_dict['loss']}")
            if cfg['model']['only_load_cnn']:  # False
                backbone_state_dict = self.model.state_dict()
                backbone_state_dict.update({
                    k: v for k, v in state_dict['model_state_dict'].items() if 'backbone' in k
                })
                self.model.load_state_dict(backbone_state_dict)
            else:
                self.model.load_state_dict(state_dict['model_state_dict'])
                self.iter_num = state_dict['iter_num']
        
        self.model = self.model.to(self.device)
        if self.cfg.optimize.parallel:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, 
                device_ids=[self.local_rank], 
                output_device=self.local_rank,
                find_unused_parameters=True
            )
            
        assert len(cfg['eval']['goals']) > 0
        
        self.inp_goals = list(zip(cfg['eval']['goals'], cfg['eval']['env_id'])) * cfg['eval']['goal_ratio']
        
        eval_goals = list(set(self.cfg['eval']['goals']))
        print(f"[Prompt] [yellow]Candidate evaluation goals: {eval_goals}")
        
        self.eval_metric = EvalMetric(eval_goals, max_ep_len=self.cfg['eval']['max_ep_len'])
        # Parallel Evaluation
        self.parallel_eval = ParallelEval(
            model=self.model, 
            embedding_dict=self.embedding_dict,
            envs=self.cfg['eval']['envs'], 
            resolution=self.cfg['simulator']['resolution'],
            max_ep_len=self.cfg['eval']['max_ep_len'], 
            num_workers=self.cfg['eval']['num_workers'],
            device=self.device, 
            fps=self.cfg['eval']['fps'], 
            cfg=self.cfg,
        )

    def run(self):
        self.eval_metric.reset()
        gnames, eval_results = self.parallel_eval.step(0, self.inp_goals)
        for goal_name, eval_result in zip(gnames, eval_results):
            self.eval_metric.add(goal_name, eval_result)
        metric_result = self.eval_metric.precision(k=3)
        
        fig_columns = ['timestep', 'success', 'goal']
        fig_data = []
        for _goal_name, _metric in metric_result.items():
            print(f"goal: {_goal_name}, pricision: {_metric['precision']}, pos: {_metric['pos']}, neg: {_metric['neg']}, tot: {_metric['tot']}, success: {_metric['success']}")

            for fig_t, fig_suc in enumerate(_metric['suc_per_step']):
                fig_data.append((fig_t, fig_suc, _goal_name))
            
        fig_df = pd.DataFrame(fig_data, columns=fig_columns)
        print(fig_df.head())
        g = sns.relplot(x='timestep', y='success', hue='goal', kind='line', data=fig_df)
        g.savefig('accumulated_success.png', dpi = 300)
    

@hydra.main(config_path="configs", config_name="defaults")
def main(cfg):
    if cfg.optimize.parallel:  # False
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.distributed.init_process_group(backend='nccl')
    else:
        local_rank = 0
        
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    from rich.console import Console
    from rich.syntax import Syntax
    console = Console()
    syntax = Syntax(str(cfg), "json", theme="monokai", line_numbers=True)
    console.print(cfg)
    
    trainer = Trainer(cfg, device=device, local_rank=local_rank) 
    trainer.run()

if __name__ == "__main__":
    main()
    
