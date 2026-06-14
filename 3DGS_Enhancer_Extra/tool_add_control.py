import sys
import os

# assert len(sys.argv) == 3, 'Args are wrong.'

# input_path = sys.argv[1]
# output_path = sys.argv[2]

cfg = "init_raw"

if cfg == "init":
    input_path = "/workspace/results_train/run1_relpose_1229_1490set_resume2/run1_relpose_1229_1490set_resume2/checkpoints/step21000.ckpt"
    # output_path = f"/workspace/results_train/controlnet_base/{cfg}.ckpt"
elif cfg == "init_raw":
    input_path = "/workspace/DynamiCrafter/checkpoints/dynamicrafter_512_interp_v1/model.ckpt"
output_path = f"/workspace/results_train/controlnet_base/{cfg}.ckpt"

assert os.path.exists(input_path), 'Input model does not exist.'
# assert not os.path.exists(output_path), 'Output filename already exists.'
if os.path.exists(output_path):
    os.remove(output_path)
    print("[INFO] Remove already existing output model.", output_path)
assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

import torch
from omegaconf import OmegaConf
# from cldm.model import create_model
from utils.utils import instantiate_from_config

def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    return model

def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]

model = create_model(config_path=f'configs/control/{cfg}.yaml')

pretrained_weights = torch.load(input_path)
if 'state_dict' in pretrained_weights:
    pretrained_weights = pretrained_weights['state_dict']

scratch_dict = model.state_dict()

target_dict = {}
for k in scratch_dict.keys():
    is_control, name = get_node_name(k, 'control_')
    if is_control:
        copy_k = 'model.diffusion_' + name
    else:
        copy_k = k
    if copy_k in pretrained_weights:
        target_dict[k] = pretrained_weights[copy_k].clone()
    else:
        target_dict[k] = scratch_dict[k].clone()
        print(f'These weights are newly added: {k}')

model.load_state_dict(target_dict, strict=True)
torch.save(model.state_dict(), output_path)
print('Done.')