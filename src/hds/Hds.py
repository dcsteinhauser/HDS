import functools

from brax import envs
from brax.training.types import Params
from brax.training.types import PRNGKey
import flax
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from IPython.display import clear_output
from functools import partial

from src.policy.DeterministicPolicy import DeterministicPolicy
from src.policy.StochasticPolicy import StochasticPolicy
from src.env.Pendulum import State
import time 
from flax import serialization
import pickle

@flax.struct.dataclass
class TrainState:
    policy_model: DeterministicPolicy
    policy_params: Params
    optimizer_state: optax.OptState
    optimizer: optax.GradientTransformation

def generate_trajectory(environment, train_state: TrainState, trajectory_length: int, num_samples, prng_keys: PRNGKey):
    
    states = jnp.zeros((num_samples, trajectory_length, environment.observation_size))
    actions = jnp.zeros((num_samples, trajectory_length, environment.action_size))
    state: State = environment.reset(prng_keys)
    
    for i in range(trajectory_length):
        action = train_state.policy_model.apply(train_state.policy_params, state.obs)
        next_state = environment.step(state, action)
        states=states.at[:,i].set(state.obs)
        actions=actions.at[:,i].set(action)
        state = next_state

    return states, actions


def generate_trajectory_parallel(environment, train_state: TrainState, trajectory_length: int, num_samples: int, prng_keys: PRNGKey):

    def step_trajectory(state, rng_key):
        action = train_state.policy_model.apply(train_state.policy_params, state.obs, rng_key)
        next_state = environment.step(state, action)
        return next_state, (next_state.obs, action,next_state.reward)

    state: State = jax.vmap(environment.reset)(prng_keys)

    keys = jax.vmap(jax.random.split, in_axes=(0,None))(prng_keys, trajectory_length)
    _, (updatedstates, updatedactions,reward) = jax.vmap(jax.lax.scan, in_axes=(None,0))(step_trajectory, state, xs=keys)
    states=jax.numpy.reshape(updatedstates, (num_samples, trajectory_length, environment.observation_size))
    actions=jax.numpy.reshape(updatedactions, (num_samples, trajectory_length, environment.action_size))
    totalreward=jnp.sum(reward, axis=0)
    return states, actions, totalreward


@partial(jax.vmap,in_axes=(None,0,0,None),out_axes=0,axis_name="batch")
@partial(jax.jit, static_argnums=(0,))
def fo_update_action_sequence(environment, actions, prng_key, alpha_a):

    def total_reward(environment, actions, prng_key):
        
        def reward_step(states, action):
            return environment.step(states, action), states.reward
        
        initial_states = environment.reset(prng_key)
        _, rewards = jax.lax.scan(f=reward_step, init=initial_states, xs=actions)
        return jnp.sum(rewards, axis=0)
    
    grad =  jax.grad(total_reward, argnums=1)(environment, actions, prng_key)
    improved_action_sequence = actions + alpha_a * grad
    return improved_action_sequence


def update_policy(states, actions, train_state, rng_key):
    params = train_state.policy_params
    policy_model = train_state.policy_model
    optimizer_state = train_state.optimizer_state
    optimizer = train_state.optimizer
    
    policy_output_fn = partial(policy_model.apply, rng_key=rng_key)
    loss_fn = lambda params, states, actions: 0.5*optax.losses.squared_error(policy_output_fn(params,states), actions).mean()
    value,grad = jax.value_and_grad(loss_fn)(params, states, actions)
    updates, optimizer_state = optimizer.update(grad, optimizer_state)
    new_params = optax.apply_updates(params, updates)
    train_state = train_state.replace(policy_params=new_params, optimizer_state=optimizer_state)

    return value, train_state

def train(
    env,
    trajectory_length: int,
    num_samples: int,
    epochs: int,
    alpha_a: float,
    progress_fn=None):

    # get a random key
    key = jax.random.PRNGKey(0)
    new_key, subkey = jax.random.split(key)

    # Define the policy and initialize it
    observation_size = int(env.observation_size)
    action_size = int(env.action_size)
    policy_model = StochasticPolicy(observation_size=observation_size,action_size= action_size)
    policy_params = policy_model.init(key, jnp.ones((observation_size,)), subkey)
    t0= time.time()

    # Define the optimizer
    scheduler = optax.exponential_decay(
        init_value=1e-5,
        transition_steps=1000,
        decay_rate=0.99)
    
    # Combining gradient transforms using `optax.chain`.
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Clip by the gradient by the global norm.
        optax.scale_by_adam(),  # Use the updates from adam.
        optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
        # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
        optax.scale(-1.0))
    
    optimizer_state = optimizer.init(policy_params)

    # Initialize the training state
    train_state = TrainState(
        policy_model=policy_model, 
        policy_params=policy_params,
        optimizer_state=optimizer_state,
        optimizer=optimizer)
    
    # Initialize a nonbatched env
    non_batched_env = env 
    # Wrap the environment to allow vmapping
    environment = envs.training.wrap(env, episode_length=trajectory_length,)
    
    # 1. run m episodes of the environment using the policy, of length trajectory_length
    # 2. collect the states and actions encountered in each episode
    # 3. for each episode initialize an array which has the sequence of actions taken by the policy
    # 4. rerun the environment, using the array of actions as input, and calculate the total reward
    # 5. calculate the gradient of the total reward with respect to the array of aprng_key[0]
    x_data = []
    y_data = []
    for i in range(epochs):
        # update rng keys
        key1, key2, key3 = jax.random.split(new_key, num=3)
        new_key = key1
        subkeys = jax.random.split(key2, num_samples)
        
        # generate trajectories
        trajectories = generate_trajectory_parallel(non_batched_env, train_state, trajectory_length, num_samples, subkeys)
        total_reward = trajectories[2]
        trajectories = trajectories[:2]
        
        # output progress
        progress_fn(x_data,y_data,i,jnp.mean(total_reward))
        
        # update action sequence
        states, actions = trajectories[0], fo_update_action_sequence(non_batched_env, trajectories[1], subkeys, alpha_a)
        
        # supervised learning
        for j in range(20):
            for state_sequence, action_sequence in zip(states, actions):
                value,train_state= update_policy(state_sequence, action_sequence, train_state, key3)
                key4, key5 = jax.random.split(key3)
                key3 = key4
            print("big epoch:",i,"small epoch:",j,"Loss",value)
            if(value<1e-4 or value == jnp.nan):
                break
       
       # checkpoint
        if i % 10 == 0:    
            params = serialization.to_state_dict(train_state.policy_params)
            with open("params.pkl", "wb") as f:
                pickle.dump(params, f)

    return functools.partial(train_state.policy_model.apply, variables=train_state.policy_params)

            