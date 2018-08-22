import numpy as np

def run_game(env, agent):
    '''Run game with agent'''

    # getting brain name
    brain_name = env.brain_names[0]
    
    # reset the environment
    env_info = env.reset(train_mode=False)[brain_name]

    # getting initial frame
    frame_0 = env_info.vector_observations[0]
    state = np.zeros(shape=(len(frame_0), 4))
    state[:,0] = frame_0

    # building initial state of 4 frames, by moving right 3 times
    state = np.zeros(shape=(len(frame_0), 4))
    state[:,0] = frame_0
    for j in range(1, 4):
        next_env_info = env.step(3)[brain_name]
        next_frame = next_env_info.vector_observations[0]
        state[:, j] = next_frame

    # initialize the score
    score = 0
    
    # while episode not over
    while True:
        
        # retrieving action from agent
        action = agent.act(state.flatten('F'))

        # taking action
        next_env_info = env.step(int(action))[brain_name]

        # retriving env_info
        next_frame = next_env_info.vector_observations[0]

        # building next state
        next_state = np.c_[state[:, 1:4], next_frame.reshape((-1, 1))]

        # retrieving reward and done
        reward = next_env_info.rewards[0]
        done = next_env_info.local_done[0]

        score += reward

        # breaking if end of episode
        if done:
            break

        state = next_state

    return score
