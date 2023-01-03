import gc
import logging
import os
import tensorflow as tf

from chess.pgn import Game
from multiprocessing import Process
from ChessAI.agent import Agent
from ChessAI.agent import RandomAgent
from ChessAI.environment import Board
from ChessAI.learn import TD_search
from keras.models import load_model

tf.autograph.set_verbosity(0)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

print("gpus:", tf.config.list_physical_devices('GPU'))

logging.basicConfig(filename="training.log", level=logging.INFO)


def train(i):
    agent = Agent(network='big')
    gc.collect()
    board = Board(agent)
    agent.model = load_model('agent_model.h5')
    R = TD_search(env=board, agent=agent, memsize=12000)
    finalboard = R.learn(iters=25, maxiter=120 - (t * 2), c=5)
    print(finalboard)
    R.agent.model.save('agent_model.h5')
    logging.info(f"{i} model variations done \n {finalboard} \n \n")


if __name__ == '__main__':
    t = 50

    agent = Agent(network='big', lr=0.5)

    if not os.path.isfile('agent_model.h5'):
        print("No model detected. Creating new model.")
        logging.info("No model detected. Creating new model.")
        ragent = RandomAgent()
        board = Board(ragent, capture_reward_factor=1)
        R = TD_search(env=board, agent=agent, memsize=6000)
        finalboard = R.learn(iters=100)
        logging.info("New model created that is trained to capture.")
        R.agent.model.save('agent_model.h5')
        logging.info("model saved as agent_model.h5")
        pgn = Game.from_board(R.env.board)
        with open(f"rlc_pgn({i})", "w") as log:
            log.write(str(pgn))

    for i in range(5, t):
        p = Process(target=train, args=(i,))
        p.start()
        p.join()
