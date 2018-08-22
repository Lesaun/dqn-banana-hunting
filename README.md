[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Udacity's Deep Reinforcement ND Project 1: Navigation

### Program Details

Trains an agent to collect yellow bananas and avoid purple bananas.

![Trained Agent][image1]

#### State and action space details

The environment provides the state as a 37 dimension vector containing the agent's velocity and a ray-based perception of objects around the agent's forward direction. The reward provided by the environment is +1 for collecting a yellow banana and -1 for a purple banana.

The agent returns an integer in [0, 3] representing the following directions:

- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The environment is considered solved when the average culmative reward over 100 consecutive episodes is above 13. The current agent solves the environment after around 550 episodes.

#### Implementation details

Uses Double DQN with 3 layer FC network. See Report.md for more details.

### Getting Started

1. Download the environment for your operating system below.
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

2. Extract the contents into banana_app/

3. Install [anaconda](https://www.anaconda.com/download/)

4. Install [pytorch](https://pytorch.org/#pip-install-pytorch) and [unityagents](https://pypi.org/project/unityagents/)

### Instructions

Run either main.py or use navigation.ipynb to run environment on existing model or retrain.z