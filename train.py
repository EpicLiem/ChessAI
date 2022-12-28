import gc
import logging
import os

from RLC.real_chess.agent import Agent
from RLC.real_chess.agent import RandomAgent
from RLC.real_chess.environment import Board
from RLC.real_chess.learn import TD_search
from keras.models import load_model

logging.basicConfig(filename="training.log", level=logging.INFO)

t = 50

agent = Agent(network='big',lr=0.5)

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

for i in range(t):
    gc.collect()
    board = Board(agent)
    agent.model = load_model('agent_model.h5')
    R = TD_search(env=board, agent=agent, memsize=6000)
    finalboard = R.learn(iters=25, maxiter=120-(t*2), c=5)
    print(finalboard)
    R.agent.model.save('agent_model.h5')
    logging.info(f"{i} model variations done \n {finalboard} \n \n")




