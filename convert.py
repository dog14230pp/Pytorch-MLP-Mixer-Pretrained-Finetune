import sys
sys.path.append("..")
import json
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

from model.MlpMixer import MLPMixer


jax_files = {
    'Mixer-B_16': 'jax_weights/Mixer-B_16.npz',
    'Mixer-L_16': 'jax_weights/Mixer-L_16.npz',
    'Mixer-B_16_imagenet1k': 'jax_weights/Mixer-B_16_imagenet1k.npz',
    'Mixer-L_16_imagenet1k': 'jax_weights/Mixer-L_16_imagenet1k.npz',
}


def jax_to_pytorch(k):
    k = k.replace('LayerNorm_0', 'norm1')
    k = k.replace('LayerNorm_1', 'norm2')
    k = k.replace('token_mixing/Dense_0', 'dense_token1')
    k = k.replace('token_mixing/Dense_1', 'dense_token2')
    k = k.replace('channel_mixing/Dense_0', 'dense_channel1')
    k = k.replace('channel_mixing/Dense_1', 'dense_channel2')
    k = k.replace('stem', 'conv')
    k = k.replace('kernel', 'weight')
    k = k.replace('scale', 'weight')
    k = k.replace('pre_head_layer_norm', 'prenorm')
    k = k.replace('/', '.')
    k = k.replace('MixerBlock_', 'mixerblocks.')
    k = k.lower()
    return k

def convert(npz, state_dict):
    new_state_dict = {}
    pytorch_k2v = {jax_to_pytorch(k): v for k, v in npz.items()}
    for pytorch_k, pytorch_v in state_dict.items():
        if pytorch_k not in pytorch_k2v:
            assert False
        v = pytorch_k2v[pytorch_k]
        v = torch.from_numpy(v)
        
        # Sizing
        if '.weight' in pytorch_k:
            if len(pytorch_v.shape) == 2:
                v = v.transpose(1, 0)
            if len(pytorch_v.shape) == 4:
                v = v.permute(3, 2, 0, 1)
        new_state_dict[pytorch_k] = v
    return new_state_dict


def check_model(model, name):
    '''
        If you want to check whether the model weigths are sucessfully loaded, please download the related data from the Repo: https://github.com/lukemelas/PyTorch-Pretrained-ViT/tree/master/examples
    '''
    model.eval()
    img = Image.open('simple/img.jpg')
    img = transforms.Compose([transforms.Resize(model.image_size), transforms.ToTensor(), transforms.Normalize(0.5, 0.5), transforms.Resize((224, 224)),])(img).unsqueeze(0)
    print(img.shape)
    if 'imagenet1k' in name:
        labels_file = 'simple/labels_map.txt' 
        labels_map = json.load(open(labels_file))
        labels_map = [labels_map[str(i)] for i in range(1000)]
        print('-----\nShould be index 388 (panda) w/ high probability:')
    else:
        print('Bot checked!')
        return # labels_map = open('../examples/simple/labels_map_21k.txt').read().splitlines()
    with torch.no_grad():
        outputs = model(img).squeeze(0)
    for idx in torch.topk(outputs, k=3).indices.tolist():
        prob = torch.softmax(outputs, -1)[idx].item()
        print('[{idx}] {label:<75} ({p:.2f}%)'.format(idx=idx, label=labels_map[idx], p=prob*100))


for name, filename in jax_files.items():
    
    # Load Jax weights
    npz = np.load(filename)

    # Load PyTorch model
    arch = name[6:]
    print('arch: {}'.format(arch))
    model = MLPMixer(arch=arch)
    
    # Convert weights
    new_state_dict = convert(npz, model.state_dict())

    # Print the model arch.
    print(model)

    # Load into model and test
    model.load_state_dict(new_state_dict)
    # print(f'Checking: {name}')
    # check_model(model, name)

    # Save weights
    new_filename = f'converted_weights/{name}.pth'
    torch.save(new_state_dict, new_filename, _use_new_zipfile_serialization=False)
    print(f"Converted {filename} and saved to {new_filename}")

