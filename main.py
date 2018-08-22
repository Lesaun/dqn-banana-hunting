import random

from unityagents import UnityEnvironment
import numpy as np
import torch

from agent import Agent
from train import train
from run_game import run_game

# loading the environment
env = UnityEnvironment(file_name='banana_app/Banana.app', seed=random.randint(0,10000))

# getting state and action space
env_info = env.reset(train_mode=True)[env.brain_names[0]]
state_size = len(env_info.vector_observations[0]) * 4
action_size = env.brains[env.brain_names[0]].vector_action_space_size

if input('Retrain agent? yes/[no]') == 'yes':
    # training new agent on model and save weights
    agent = Agent(state_size, action_size)
    scores = train(env, agent, n_episodes = 1000)
    torch.save(agent.Qθ.state_dict(), 'model_weights.pth')
else:
    # running game with previously trained agent
    model_state = torch.load('model_weights.pth')
    agent = Agent(state_size, action_size, model_state = model_state)
    score = run_game(env, agent)
    print('Score ' + str(score))

env.close()
