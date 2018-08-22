import copy
from collections import namedtuple, deque
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Network

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
LR = 5e-4
UPDATE_EVERY = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, model_state = None):
        self.state_size = state_size
        self.action_size = action_size

        self.Qθ = Network(state_size, action_size).to(device)
        self.Qθbar = copy.deepcopy(self.Qθ).to(device)

        self.optimizer = optim.Adam(self.Qθ.parameters(), lr=LR)
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)

        self.t_step = 0

        if model_state:
            self.Qθ.load_state_dict(model_state)
            self.Qθbar.load_state_dict(model_state)

    def step(self, state, action, reward, next_state, done):
        """Adds new experience to memory and updates network"""
        self.memory.add(state, action, reward, next_state, done)
        
        # update networks every UPDATE_EVERY steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.Qθ.eval()

        with torch.no_grad():
            action_values = self.Qθ(state)

        self.Qθ.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Sample experiences and optimize policy."""
        states, actions, rewards, next_states, dones = experiences

        # getting max actions from Qtheta network
        a_max = self.Qθ(next_states).detach().max(1)[1].unsqueeze(1)

        # getting q values from QthetaBar for each max action
        q_target = self.Qθbar(next_states).gather(1, a_max)

        # setting state target
        target = rewards + (gamma * q_target * (1 - dones))

        # evaluating current state q-value on action taken
        current = self.Qθ(states).gather(1, actions)

        # calculating loss
        loss = F.mse_loss(current, target)

        # optimizing
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # updating parameters
        self.soft_update(self.Qθ, self.Qθbar, TAU)                     

    def soft_update(self, Qθ, Qθbar, tau):
        """Soft copies parameters from Qθ to Qθbar networks"""
        for Qθbar_param, Qθ_param in zip(Qθbar.parameters(), Qθ.parameters()):
            Qθbar_param.data.copy_(tau * Qθ_param.data + (1.0 - tau) * Qθbar_param.data)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size):
        self.action_size = action_size
        self.memory = deque(maxlen = buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)