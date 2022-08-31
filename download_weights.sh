#!/usr/bin/ bash

if [ ! -d "jax_weights/" ]
then
    mkdir "jax_weights/"
    echo "Folder jax_weights created"
fi

if [ ! -d "converted_weights/" ]
then
    mkdir "converted_weights/"
    echo "Folder converted_weights created"
fi

# B/16
wget https://storage.googleapis.com/mixer_models/imagenet21k/Mixer-B_16.npz -O jax_weights/Mixer-B_16.npz

# L/16
wget https://storage.googleapis.com/mixer_models/imagenet21k/Mixer-L_16.npz -O jax_weights/Mixer-L_16.npz

# B/16 Finetune on imagenet1k
wget https://storage.googleapis.com/mixer_models/imagenet1k/Mixer-B_16.npz -O jax_weights/Mixer-B_16_imagenet1k.npz

# L/16 Finetune on imagenet1k
wget https://storage.googleapis.com/mixer_models/imagenet1k/Mixer-L_16.npz -O jax_weights/Mixer-L_16_imagenet1k.npz
