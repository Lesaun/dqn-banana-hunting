import numpy as np

def train(env, agent, n_episodes = 1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    '''Train the agent on environment'''
    solved = False

    # getting brain name
    brain_name = env.brain_names[0]

    # initializing scores over all episodes
    scores = []

    # initializing epsilon
    eps = eps_start

    # for each episode:
    for i_episode in range(1, n_episodes + 1):

        # resetting environment
        env_info = env.reset(train_mode=True)[brain_name]

        # getting initial state
        frame_0 = env_info.vector_observations[0]
        #state = np.expand_dims(frame_0, 0)
        state = np.zeros(shape=(len(frame_0), 4))
        state[:,0] = frame_0

        # moving right 4 frames to build initial temporal state
        for j in range(1, 4):
            next_env_info = env.step(3)[brain_name]
            next_frame = next_env_info.vector_observations[0]
            state[:, j] = next_frame

        # initializing episode score
        score = 0

        while True:

            # getting action from agent
            action = agent.act(state.flatten('F'), eps)#.flatten('F'), eps)

            # taking action
            next_env_info = env.step(int(action))[brain_name]

            # retriving env_info
            next_frame = next_env_info.vector_observations[0]

            # building next state
            next_state = np.c_[state[:, 1:4], next_frame.reshape((-1, 1))]
            
            # retrieving reward and done
            reward = next_env_info.rewards[0]
            done = next_env_info.local_done[0]

            # stepping agent forward
            agent.step(state.flatten('F'), action, reward, next_state.flatten('F'), done)

            score += reward

            # breaking if end of episode
            if done:
                break

            state = next_state

        # decaying epsilon
        eps = max(eps_end, eps_decay * eps)

        scores.append(score)

        # printing status
        if i_episode % 10 == 0:
            print("Episode", i_episode - 10, "to", i_episode, "scores: ",
                  " ".join([ "%02d" % s for s in scores[-10:]]))

        # print episode where environment is solved
        if np.mean(scores[-100:]) > 13 and not solved:
            solved = True
            print("Solved on episode: " + str(i_episode))

    return scores
