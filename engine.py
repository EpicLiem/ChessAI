import time

import numpy as np
import gc
import lib

import chess
from keras.models import load_model

model = load_model('agent_model.h5')

board = chess.Board()

print(lib.predictmove(model, board))

