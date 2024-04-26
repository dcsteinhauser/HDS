import functools
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union

from absl import logging
from brax import base
from brax import envs
from brax.training import acting
from brax.training import gradients
from brax.training import pmap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.apg import networks as apg_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
from brax.v1 import envs as envs_v1
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
import optax

from ..policy.DeterministicPolicy import DeterministicPolicy


@flax.struct.dataclass
class TrainState:
    environment: envs.Env
    policy_model: DeterministicPolicy
    policy_params: Params
    optimizer: optax.OptState
    state_action_dict: dict

    def clear_state_action_dict(self):
        self.state_action_dict = {}


def loss_fn_policy(policy_outputs, targets):
    return jnp.mean(0.5*jnp.square(policy_outputs - targets))
    
def generate_trajectory(environment, train_state, trajectory_length):
    states = jnp.zeros((trajectory_length, environment.observation_size))
    actions = jnp.zeros((trajectory_length, environment.action_size))

    


def train(
    environment: envs.Env,
    trajectory_length: int,
    epochs: int):

    # Define the policy and initialize it
    observation_size = environment.observation_size
    action_size = environment.action_size

    policy_model = DeterministicPolicy(observation_size, action_size)
    policy_params = policy_model.init(jax.random.PRNGKey(0), jnp.ones((observation_size,)))

    # Define the optimizer
    optimizer_state = optax.adam().init(policy_params)

    # Initialize the training state
    train_state = TrainState(
        environment=environment,
        policy_model=policy_model, 
        policy_params=policy_params,
        optimizer=optimizer_state, 
        state_action_dict={})
    
    # 1. run m episodes of the environment using the policy, of length trajectory_length
    # 2. collect the states and actions encountered in each episode
    # 3. for each episode initialize an array which has the sequence of actions taken by the policy
    # 4. rerun the environment, using the array of actions as input, and calculate the total reward
    # 5. calculate the gradient of the total reward with respect to the array of actions
    # 6. update the array of actions
    # 7. perform supervised learning on the policy using the array of actions

    for _ in epochs:
