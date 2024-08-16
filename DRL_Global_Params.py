STATE_SPACE = 28
ACTION_SPACE = 3

ACTION_LOW = -1
ACTION_HIGH = 1

GAMMA = 0.9995
TAU = 1e-3
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 0.9

MEMORY_LEN = 10000
MEMORY_THRESH = 500
BATCH_SIZE = 200

LR_DQN = 5e-4

LEARN_AFTER = MEMORY_THRESH
LEARN_EVERY = 3
UPDATE_EVERY = 9

COST = 3e-4
CAPITAL = 100
NEG_MUL = 2
