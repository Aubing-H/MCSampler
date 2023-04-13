
import torch

from src.child import ChildSampler

def test_child_model(cfg):
    device = torch.cuda.set_device(cfg['device'])
    bs = 1
    obs = {
        'rgb': torch.zeros([bs, 3, 480, 640], dtype=torch.float32, device=device),
        'voxels': torch.ones([bs, 3, 2, 2], dtype=torch.long, device=device),
        'compass': torch.zeros([bs, 2], dtype=torch.float32, device=device),
        'gps': torch.zeros([bs, 3], dtype=torch.float32, device=device),
        'biome': torch.ones([bs], dtype=torch.long, device=device),
        'prev_action': torch.zeros([bs, 8], dtype=torch.long, device=device),
    }
    child = ChildSampler(cfg, device=device)
    action = child._get_action('log', obs)
    print('Action: ', action)
