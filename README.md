## MLP Mixer - Pytorch with Finetune Implementation

This is the repositories that change the weights from Jax (Google Research) to Pytorch. And I also provide the code that people can easliy finetune the MLP-Mixer by themselves.

## Usage (Training and Testing)

```
main.py [-h] [-mode {train,test}]
             [-arch {B_16,L_16,Mixer-B_16_imagenet1k,Mixer-L_16_imagenet1k}]
             [-cp CP] 
             [-dataset {cifar10,cifar100}] 
             [-E EPOCHS]
             [-LR LEARNING_RATE] 
             [-BS BATCH_SIZE] 
             [-TBS TEST_BATCH_SIZE]
             [-pretrained PRETRAINED] 
             [-finetune FINETUNE]
```

Here are the explanation of the arguments:

* ```mode```: A string. Training or testing the MLP-Mixer. Input can be [ train | test ].
* ```arch```: A string. Specify the architecture of MLP-Mixer. Input can be [ B_16 | L_16 | Mixer-B_16_imagenet1k | Mixer-L_16_imagenet1k ].
* ```cp```: A string. The path of the checkpoints. For finetuning, then it will be the pretrained weights. For testing, then it will be the trained weights.
* ```dataset```: A string. Specify the datasets. Input can be [ cifar10 | cifar100 ].
* ```E```: An int. Specify the training epochs for training MLP-Mixer.
* ```LR```: A float. Specify the learning rate for training MLP-Mixer.
* ```BS```: An int. Specify the batch size on training dataset for training or testing MLP-Mixer.
* ```TBS```: An int. Specify the batch size on testing dataset for training or testing MLP-Mixer.
* ```pretrained```: A boolean. Specify whether to use pretrained model or not.
* ```finetune```: A boolean. Specify whether to finetune the model or not.

You can train MLP-Mixer by using the script (with pretrained && finetune) below (Take Cifar10 as example):

```python
python main.py -mode train -arch B_16 -cp [please_input_the_path_of_the_weights_here] -dataset cifar10 -E 50 -LR 0.001 -BS 64 -pretrained True -finetune True
```

You can test MLP-Mixer by using the script below (Take Cifar10 as example):

```python
 python main.py -mode test -arch B_16 -cp [please_input_the_path_of_the_weights_here] -dataset cifar10 -BS 32 -TBS 32
```


## Download and Convert the Weights

You can download the weights by running the script below:

```
sh download_weights.sh
```

After downloading the pretrained weights, then you can convert the weights into Pytorch type:

```
python convert.py
```

If you have other demands, please the github from Google Research: https://github.com/google-research/vision_transformer

## Finetune Result Example
| Dataset  | Training Data Acc. | Testing Data Acc. |
| ------------- | ------------- | ------------- |
| Cifar10 | 99.91% | 92.97% |


## Note
The project is really flexible, you can add your own dataset, data augumentation techniques or other crazy features by yourself. And if you have done anything interesting, please let me know by creating the github issue. Thanks a lot.

## Code Reference
1. For convert the weights: https://github.com/lukemelas/PyTorch-Pretrained-ViT
2. For reshaping the shape: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
3. For MLP-Mixer: https://github.com/google-research/vision_transformer