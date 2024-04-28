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
from ..env.Pendulum import State


@flax.struct.dataclass
class TrainState:
    environment: envs.Env
    policy_model: DeterministicPolicy
    policy_params: Params
    optimizer: optax.OptState

    def clear_state_action_dict(self):
        self.state_action_dict = {}
    
@jax.jit
def generate_trajectory(environment: envs.Env, train_state: TrainState, trajectory_length: int, prng_key: PRNGKey):
    states = jnp.zeros((trajectory_length, environment.observation_size))
    actions = jnp.zeros((trajectory_length, environment.action_size))

    state: State = environment.reset(prng_key)
    for i in range(trajectory_length):
        action = train_state.policy_model.apply(train_state.policy_params, state.obs)
        next_state = environment.step(state, action)
        states[i] = state.obs
        actions[i] = action
        state = next_state

    return states, actions


@jax.jit
def fo_update_action_sequence(environment, states, actions, alpha_a):
    
    @jax.jit
    def total_reward(environment, states, actions):
        state: State = environment.reset()
        total_reward = jnp.zeros(1);
    
        for i in range(states.shape[0]):
            action = actions[i]
            next_state = environment.step(state, action)
            reward += next_state.reward
            state = next_state

        return total_reward
    
    progress, grad = jax.value_and_grad(fun=total_reward, argnums=1)(environment, states, actions)
    improved_action_sequence = actions + alpha_a * grad
    return improved_action_sequence

@jax.jit
def update_policy(states, actions, train_state):

    @jax.jit
    def loss_fn_policy(policy_outputs, targets):
        return jnp.mean(0.5*jnp.square(policy_outputs - targets))

    for state, action in zip(states, actions):
        policy_outputs = train_state.policy_model.apply(train_state.policy_params, state)
        loss = loss_fn_policy(policy_outputs, action)
        grad = jax.grad(loss)(train_state.policy_params)
        optimizer_state = optax.adam().update(grad, optimizer_state)
        train_state = train_state.replace(optimizer=optimizer_state)
        params = optax.apply_updates(train_state.policy_params, optimizer_state)
        train_state = train_state.replace(policy_params=params)

    return train_state;

def train(
    environment: envs.Env,
    trajectory_length: int,
    num_samples: int,
    epochs: int,
    alpha_a: float,
    progress_fn=None):

    # get a random key
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)

    # Define the policy and initialize it
    observation_size = environment.observation_size
    action_size = environment.action_size

    policy_model = DeterministicPolicy(observation_size, action_size)
    policy_params = policy_model.init(subkey, jnp.ones((observation_size,)))

    # Define the optimizer
    optimizer_state = optax.adam().init(policy_params)

    # Initialize the training state
    train_state = TrainState(
        environment=environment,
        policy_model=policy_model, 
        policy_params=policy_params,
        optimizer=optimizer_state)
    
    # 1. run m episodes of the environment using the policy, of length trajectory_length
    # 2. collect the states and actions encountered in each episode
    # 3. for each episode initialize an array which has the sequence of actions taken by the policy
    # 4. rerun the environment, using the array of actions as input, and calculate the total reward
    # 5. calculate the gradient of the total reward with respect to the array of actions
    # 6. update the array of actions
    # 7. perform supervised learning on the policy using the array of actions

    for _ in epochs:
        trajectories = set()
        # 1 - 3
        for i in num_samples:
            key, subkey = jax.random.split(key)
            states, actions = generate_trajectory(environment, train_state, trajectory_length, subkey)
            trajectories.add((states, actions))

        # 4 - 6, first order update
        jax.tree_util.tree_map(lambda states, actions: (states, fo_update_action_sequence(environment, states, actions, alpha_a)), trajectories)
        
        # 7
        for _ in epochs:
            for states, actions in trajectories:
                train_state = update_policy(states, actions, train_state)
        
    
    return train_state.policy_params

            
