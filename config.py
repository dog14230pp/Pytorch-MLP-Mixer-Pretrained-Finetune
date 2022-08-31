from easydict import EasyDict

cfg = EasyDict()

# Setting hyperparameters
cfg.test_batch_size = 100
cfg.learning_rate = 1
cfg.batch_size = 512
cfg.precision = 8
cfg.decay = 0.96
cfg.epochs = 2

# Setting the folders
cfg.model_dir_mlpmixer = "model_weights/"
cfg.data_dir = "../dataset"



