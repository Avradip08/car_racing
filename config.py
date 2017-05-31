"""
For each module, put its config variables / constants in its own respective
"""

class WorldConfig():
    NAME = "CarRacing-v0"

    # POSSIBLE ACTIONS
    GAS = [0.0, 1.0, 0.0]
    BRAKE = [0.0, 0.0, 0.95]
    LEFT = [-1.0, 0.0, 0.0]
    RIGHT = [1.0, 0.0, 0.0]

    ACTIONS = [GAS, BRAKE, LEFT, RIGHT]

    NUM_ACTIONS = len(ACTIONS)

    # Each state is 3rd dimensional stack of consecutive frames
    NUM_FRAMES_IN_STATE = 5
    WORLD_SIZE = 84 #TODO: why 84?


class A3CConfig():
    NUM_THREADS = 8
    NUM_PLAY_THREADS = 6
    NUM_REPLAY_THREADS = 2

class ACNetworkConfig():
    LR_START = 0.0001

    FEATURE_DIM = 128

    SAVE_PATH = "./checkpoint/"

class ACWorkerConfig():
    T = 5
    GAMMA = 0.99
    MAX_ITERATIONS = 4 * (10 ** 6)
    BUFFER_SIZE = 10000
    BATCH_SIZE = 32
    REPLAY_START = 3000
