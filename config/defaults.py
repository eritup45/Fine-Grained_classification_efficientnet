from yacs.config import CfgNode as CN
import os

# Folder, where executing train.py, seen as root path.
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

_C = CN()
# _C.merge_from_file(os.path.join(BASE_PATH, "config.yml"))

_C.PATH = CN()
_C.MODEL = CN()
_C.DEVICE = CN()
_C.DATA = CN()

_C.DEVICE.GPU = 0 # <gpu_id>
_C.DEVICE.CUDA = True # use gpu or not

_C.PATH.TRAIN_SET = "./data/train" # <path_to_trainset>
_C.PATH.TEST_SET = "./data/test" # <path_to_testset>
_C.PATH.TRAIN_ANNOTATION = ""
_C.PATH.TEST_ANNOTATION = ""

_C.MODEL.OUTPUT_PATH = "./weights/model.pth" # <weight_output_path>
_C.MODEL.RESUME_PATH = "./weights/model.pth" # <weight_loaded>
_C.MODEL.FINAL_MODEL_PATH = ""
_C.MODEL.LR = 0.0001 # <learning_rate>
_C.MODEL.WARMUP_STEP = 500 # <LR_WARMUP_STEP>
_C.MODEL.DECAY_TYPE = "warmup_cosine" 
_C.MODEL.EPOCH = 5 # <train_epochs>

# -----------------------------------------------
# normalization parameters(suggestion)
_C.DATA.PIXEL_MEAN = [0.485, 0.456, 0.406] 
_C.DATA.PIXEL_STD = [0.229, 0.224, 0.225]

# -----------------------------------------------
# Images augmentation
_C.DATA.RESIZE = [224, 224] # picture size after resizing
_C.DATA.DEGREES = 90 # rotate picture degrees when training
_C.DATA.TRANSLATE = 0.2 # picture shifts img_width*0.2 when training

# -----------------------------------------------
# Training options
_C.DATA.NUM_WORKERS = 4 # use how many processors
_C.DATA.TRAIN_BATCH_SIZE = 32 # <train_batch_size>
_C.DATA.TEST_BATCH_SIZE = 16 # <test_batch_size>
_C.DATA.VALIDATION_PROPORTION = 0.2
_C.DATA.NUM_CLASS = 2
_C.DATA.RAND_N = 2
_C.DATA.RAND_M = 17

_C.merge_from_file(os.path.join(BASE_PATH, "config.yml"))