
ROOT_PATH = '/opt/data/private/code/reweight/'
CHECKPOINT_PATH = ROOT_PATH + 'checkpoint/'
DATA_SOURCE_PATH = '/opt/data/private/dataset/'
logDir = ROOT_PATH + 'log'

DATA_PATH = '/opt/data/private/data/' 

BATCH_SIZE = 32
MAX_EPOCHS = 100
LEARNING_RATE = 0.0002
SAVE_EPOCH = 5
XCEPTION = { 
    'img_size': (320, 320),
    'map_size': (10, 10),
    'norms': [[0.5] * 3, [0.5] * 3]
}
