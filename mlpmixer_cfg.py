# Config for MLP-Mixer
def get_base_config():
    '''
        Base MLP-Mixer config.
    '''
    return dict(
        image_size=224,
        channels=3, 
        patch_size=16, 
        dim=768, 
        depth=12, 
        num_classes=21843, 
        token_dim=384, 
        channel_dim=3072
    )

def get_b16_config():
    '''
        Returns the MLP-Mixer-B/16 configuration.
    '''
    config = get_base_config()
    return config

def get_l16_config():
    '''
        Returns the MLP-Mixer-L/16 configuration.
    '''
    config = get_base_config()
    config.update(dict(
        patches=(16, 16),
        dim=1024,
        depth=24, 
        token_dim=512, 
        channel_dim=4096
    ))
    return config

def get_b16_imagenet1k_config():
    '''
        Returns the MLP-Mixer-B/16 finetune on imagenet-1K configuration.
    '''
    config = get_base_config()
    config.update(dict(
        num_classes=1000, 
    ))
    return config

def get_l16_imagenet1k_config():
    '''
        Returns the mlp-mixer-L/16 finetune on imagenet-1K configuration.
    '''
    config = get_base_config()
    config.update(dict(
        patches=(16, 16),
        dim=1024,
        depth=24, 
        num_classes=1000, 
        token_dim=512, 
        channel_dim=4096
    ))
    return config