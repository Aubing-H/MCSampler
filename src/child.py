
import os
import numpy as np

import torch
from ray.rllib.models.torch.torch_action_dist import TorchMultiCategorical

from rich import print

from src.models.simple import SimpleNetwork
from src.utils.vision import create_backbone, resize_image

# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5'
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True


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

class ChildSampler:
    
    def __init__(self, cfg, device):
        
        self.action_space = [3, 3, 4, 11, 11, 8, 1, 1]
        self.cfg = cfg
        self.device = device
        self.prev_action = torch.tensor([0, 0, 0, 5, 5, 0, 0, 0], 
            dtype=torch.long).unsqueeze(0).to(self.device)

        #! accquire goal embeddings
        print("[Progress] [red]Computing goal embeddings using MineClip's text encoder...")
        # log, sheep, cow, pig; use mineclip to convert goal to embeddings
        self.embedding_dict = accquire_goal_embeddings(cfg['pretrains']['clip_path'], cfg['data']['filters'])
        
        backbone = create_backbone(
            name=cfg['model']['backbone_name'], 
            model_path=cfg['model']['model_path'], 
            weight_path=cfg['model']['weight_path'],
            goal_dim=cfg['model']['embed_dim'],
        )
        
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
        
        self.model = self.model.to(self.device)
        self.model.training = False
        

    def preprocess_obs(self, obs: dict):
        res_obs = {}
        rgb = torch.from_numpy(obs['rgb']).unsqueeze(0).to(device=self.device, 
            dtype=torch.float32)  # remove permute(0, 3, 1, 2)
        res_obs['rgb'] = rgb
        res_obs['voxels'] = torch.from_numpy(obs['voxels']["block_meta"]
            ).reshape(-1).unsqueeze(0).to(device=self.device, dtype=torch.long)
        res_obs['compass'] = torch.from_numpy(np.concatenate([
            obs["location_stats"]["pitch"], obs["location_stats"]["yaw"]
            ])).unsqueeze(0).to(device=self.device, dtype=torch.float32)
        res_obs['gps'] = torch.from_numpy(obs["location_stats"]["pos"] / 
            np.array([1000., 100., 1000.])).unsqueeze(0).to(device=self.device, 
                                                           dtype=torch.float32)
        res_obs['biome'] = torch.from_numpy(obs["location_stats"]["biome_id"]
            ).unsqueeze(0).to(device=self.device, dtype=torch.long)
        res_obs['prev_action'] = self.prev_action
        return res_obs
    
    def get_action(self, goal, obs):
        states = self.preprocess_obs(obs)
        return self._get_action(goal, states)

    def _get_action(self, goal, obs, horizon=20):
        goals = torch.from_numpy(self.embedding_dict[goal]).to(device=self.device)
        get_action = self.model.module.get_action if hasattr(self.model, 'module') else self.model.get_action
        horizons = torch.tensor([horizon], dtype=torch.long).to(self.device)
        # print('goals shape: {}'.format(goals.shape))
        action_preds, mid_info = get_action(
            goals=goals, 
            states=obs, 
            horizons=horizons, 
        )
        action_preds = action_preds[:, -1]
        action_space = self.model.module.action_space if hasattr(self.model, 'module') else self.model.action_space
        action_dist = TorchMultiCategorical(action_preds,  None, action_space)
        action = action_dist.sample()
        self.prev_action = action
        return action.squeeze(0).cpu().numpy()

    
