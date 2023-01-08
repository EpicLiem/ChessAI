import gc
import logging
import os

import chess
import tensorflow as tf

from chess.pgn import Game
from multiprocessing import Process
from ChessAI.agent import Agent
from ChessAI.agent import GreedyAgent, EngineWrapperAgent
from ChessAI.environment import Board
from ChessAI.learn import TD_search
from keras.models import load_model
import pandas as pd

tf.autograph.set_verbosity(0)

print("gpus:", tf.config.list_physical_devices('GPU'))

logging.basicConfig(filename="training.log", level=logging.INFO)


def train(i, FEN=None):
    agent = Agent(network='big')
    # agent.model = load_model('agent_model.h5')
    # agent.fix_model()
    gagent = GreedyAgent()
    board = Board(gagent, capture_reward_factor=0.1, FEN=FEN)
    R = TD_search(env=board, agent=agent, memsize=12000, batch_size=512)
    learining = R.learn(iters=100, c=5, maxiter=5)
    R.agent.model.save('agent_model.h5')
    logging.info(f"variation {i} completed")
    pgn = Game.from_board(R.env.board)
    with open(f"rlc_pgn({i})", "w") as log:
        log.write(str(pgn))


if __name__ == '__main__':
    t = 50

    agent = Agent(network='big', lr=0.5)

    if not os.path.isfile('agent_model.h5'):
        print("No model detected. Creating new model.")
        logging.info("No model detected. Creating new model.")
        gagent = EngineWrapperAgent()
        board = Board(gagent, capture_reward_factor=1)
        R = TD_search(env=board, agent=agent, memsize=6000)
        finalboard = R.learn(iters=100)
        logging.info("New model created that is trained to capture.")
        R.agent.model.save('agent_model.h5')
        logging.info("model saved as agent_model.h5")
        pgn = Game.from_board(R.env.board)
        with open("rlc_pgn(first)", "w") as log:
            log.write(str(pgn))

    print("Loading Data")
    data = pd.read_csv("data/puzzles.db", on_bad_lines='skip', header=None,
                       names=['PuzzleId', 'FEN', 'Moves', 'Rating', 'RatingDeviation', 'Popularity', 'NbPlays',
                              'Themes', 'GameUrl', 'OpeningFamily', 'OpeningVariation'])
    print("Getting puzzles")
    puzzles = data["FEN"]
    i = 0
    for puzzle in puzzles:
        print(chess.Board(fen=puzzle))
        train(i, FEN=puzzle)
        i +=1
