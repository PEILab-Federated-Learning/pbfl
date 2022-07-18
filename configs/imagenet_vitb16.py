from yacs.config import CfgNode as CN

###########################
# Config definition
###########################
_C = CN()
_C.VERSION = 1
# Directory to save the output files (like log.txt and model weights)
_C.OUTPUT_DIR = "./output"
# Path to a directory where the files were saved previously
_C.RESUME = ""
# Set seed to negative value to randomize everything
# Set seed to positive value to use a fixed seed
_C.SEED = 1234
_C.USE_CUDA = True
# Print detailed information
# E.g. trainer, dataset, and backbone
_C.VERBOSE = False

###########################
# FL
###########################
_C.FL = CN()
_C.FL.CLIENTS_TOTAL = 2
_C.FL.ROUNDS_TOTAL = 2

###########################
# Prompt learner
###########################
_C.PROMPT = CN()
_C.PROMPT.N_CTX = 16  # number of context vectors
_C.PROMPT.CTX_INIT = ""  # initialization words
_C.PROMPT.PREC = "fp16"  # fp16, fp32, amp
_C.PROMPT.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

###########################
# Input
###########################
_C.INPUT = CN()
_C.INPUT.SIZE = (224, 224)
# Mode of interpolation in resize functions
_C.INPUT.INTERPOLATION = "bicubic"
# For available choices please refer to transforms.py
_C.INPUT.TRANSFORMS = ["random_resized_crop", "random_flip", "normalize"]
# If True, tfm_train and tfm_test will be None
_C.INPUT.NO_TRANSFORM = False
# Mean and std (default: ImageNet)
_C.INPUT.PIXEL_MEAN = [0.48145466, 0.4578275, 0.40821073]
_C.INPUT.PIXEL_STD = [0.26862954, 0.26130258, 0.27577711]
# Random resized crop
_C.INPUT.RRCROP_SCALE = (0.08, 1.0)
# Cutout
_C.INPUT.CUTOUT_N = 1
_C.INPUT.CUTOUT_LEN = 16
# Gaussian noise
_C.INPUT.GN_MEAN = 0.0
_C.INPUT.GN_STD = 0.15

###########################
# Dataset
###########################
_C.DATASET = CN()
# Directory where datasets are stored
_C.DATASET.ROOT = "../../dataset"
_C.DATASET.NAME = "imagenet"
# Number of images per class
_C.DATASET.NUM_SHOTS = 2
# CIFAR-10/100-C's corruption type and intensity level
_C.DATASET.CIFAR_C_TYPE = ""
_C.DATASET.CIFAR_C_LEVEL = 1
# all, base or new
_C.DATASET.SUBSAMPLE_CLASSES = "all"

###########################
# Dataloader
###########################
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 8
# Apply transformations to an image K times (during training)
_C.DATALOADER.K_TRANSFORMS = 1
# img0 denotes image tensor without augmentation
# Useful for consistency learning
_C.DATALOADER.RETURN_IMG0 = False
# Setting for the train_x data-loader
_C.DATALOADER.TRAIN_X = CN()
_C.DATALOADER.TRAIN_X.SAMPLER = "RandomSampler"
_C.DATALOADER.TRAIN_X.BATCH_SIZE = 50
# Setting for the test data-loader
_C.DATALOADER.TEST = CN()
_C.DATALOADER.TEST.SAMPLER = "SequentialSampler"
_C.DATALOADER.TEST.BATCH_SIZE = 100

###########################
# Model
###########################
_C.MODEL = CN()
# Path to model weights
_C.MODEL.WEIGHT_PATH = "weights"
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = "ViT-B/16"
_C.MODEL.BACKBONE.PRETRAINED = True
# Definition of embedding layers
_C.MODEL.HEAD = CN()
# If none, do not construct embedding layers, the
# backbone's output will be passed to the classifier
_C.MODEL.HEAD.NAME = ""
# Structure of hidden layers (a list), e.g. [512, 512]
# If undefined, no embedding layer will be constructed
_C.MODEL.HEAD.HIDDEN_LAYERS = ()
_C.MODEL.HEAD.ACTIVATION = "relu"
_C.MODEL.HEAD.BN = True
_C.MODEL.HEAD.DROPOUT = 0.0

###########################
# Optimization
###########################
_C.OPTIM = CN()
_C.OPTIM.NAME = "sgd"
_C.OPTIM.LR = 0.002
_C.OPTIM.WEIGHT_DECAY = 5e-4
_C.OPTIM.MOMENTUM = 0.9
_C.OPTIM.SGD_DAMPNING = 0
_C.OPTIM.SGD_NESTEROV = False
_C.OPTIM.RMSPROP_ALPHA = 0.99
# The following also apply to other
# adaptive optimizers like adamw
_C.OPTIM.ADAM_BETA1 = 0.9
_C.OPTIM.ADAM_BETA2 = 0.999
# STAGED_LR allows different layers to have
# different lr, e.g. pre-trained base layers
# can be assigned a smaller lr than the new
# classification layer
_C.OPTIM.STAGED_LR = False
_C.OPTIM.NEW_LAYERS = ()
_C.OPTIM.BASE_LR_MULT = 0.1
# Learning rate scheduler
_C.OPTIM.LR_SCHEDULER = "cosine"
# -1 or 0 means the stepsize is equal to max_epoch
_C.OPTIM.STEPSIZE = (-1, )
_C.OPTIM.GAMMA = 0.1
_C.OPTIM.MAX_EPOCH = 2
# Set WARMUP_EPOCH larger than 0 to activate warmup training
_C.OPTIM.WARMUP_EPOCH = 1
# Either linear or constant
_C.OPTIM.WARMUP_TYPE = "constant"
# Constant learning rate when type=constant
_C.OPTIM.WARMUP_CONS_LR = 1e-5
# Minimum learning rate when type=linear
_C.OPTIM.WARMUP_MIN_LR = 1e-5
# Recount epoch for the next scheduler (last_epoch=-1)
# Otherwise last_epoch=warmup_epoch
_C.OPTIM.WARMUP_RECOUNT = True

###########################
# Train (local)
###########################
_C.TRAIN = CN()
# How often (epoch) to save model during training
# Set to 0 or negative value to only save the last one
_C.TRAIN.CHECKPOINT_FREQ = 0
# How often (batch) to print training information
_C.TRAIN.PRINT_FREQ = 5

###########################
# Test
###########################
_C.TEST = CN()
_C.TEST.EVALUATOR = "Classification"
_C.TEST.PER_CLASS_RESULT = False
# Compute confusion matrix, which will be saved
# to $OUTPUT_DIR/cmat.pt
_C.TEST.COMPUTE_CMAT = False
# If NO_TEST=True, no testing will be conducted
_C.TEST.NO_TEST = False
# Use test or val set for FINAL evaluation
_C.TEST.SPLIT = "test"
# Which model to test after training (last_step or best_val)
# If best_val, evaluation is done every epoch (if val data
# is unavailable, test data will be used)
_C.TEST.FINAL_MODEL = "last_step"
