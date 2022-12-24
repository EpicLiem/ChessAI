import numpy as np
import pandas as pd
import inspect
import gc

import chess
from chess.pgn import Game
import RLC
from RLC.real_chess.environment import Board
from RLC.real_chess.learn import TD_search
from RLC.real_chess.agent import Agent
from RLC.real_chess.agent import RandomAgent

from keras.models import load_model

def my_reset(*varnames):
    """
    varnames are what you want to keep
    """
    globals_ = globals()
    to_save = {v: globals_[v] for v in varnames}
    to_save['my_reset'] = my_reset  # lets keep this function by default
    del globals_
    globals().update(to_save)

t = 20

agent = Agent(network='convo',lr=0.07)
ragent = RandomAgent()
board = Board(ragent)
R = TD_search(env=board , agent=agent)
R.agent.fix_model()
R.agent.model.summary()


model = R.learn(iters=1, maxiter=30, c=1) # training agent against random moves with small maxiter


# R.agent.model.save('agent_model.h5')
print("first training step done")

for _ in range(t):
    gc.collect()
    board = Board(agent)
    agent.model = load_model('agent_model.h5')
    R = TD_search(env=board, agent=agent)
    R.learn(iters=25, maxiter=120-(t*2), c=5)
    R.agent.model.save('agent_model.h5')





