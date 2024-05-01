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

from src.policy.DeterministicPolicy import DeterministicPolicy
from src.env.Pendulum import State


@flax.struct.dataclass
class TrainState:
    policy_model: DeterministicPolicy
    policy_params: Params
    optimizer_state: optax.OptState
    optimizer: optax.GradientTransformation
    
# @jax.jit
def generate_trajectory(environment, train_state: TrainState, trajectory_length: int, num_samples, prng_keys: PRNGKey):
    
    states = jnp.zeros((num_samples, trajectory_length, environment.observation_size))
    actions = jnp.zeros((num_samples, trajectory_length, environment.action_size))
    state: State = environment.reset(prng_keys)
    
    # carry is the state, xs are the actions, init is the first state, y is the 
    # 
    for i in range(trajectory_length):
        action = train_state.policy_model.apply(train_state.policy_params, state.obs)
        next_state = environment.step(state, action)
        states=states.at[:,i].set(state.obs)
        actions=actions.at[:,i].set(action)
        state = next_state

    return states, actions

# @jax.jit
def generate_trajectory_parallel(environment, train_state: TrainState, trajectory_length: int, num_samples, prng_keys: PRNGKey):
    #@jax.jit
    def step_trajectory(state, _):
        action = train_state.policy_model.apply(train_state.policy_params, state.obs)
        next_state = environment.step(state, action)
        return next_state, (next_state.obs, action)

    state: State = environment.reset(prng_keys)
    _, (updatedstates, updatedactions) = jax.lax.scan(step_trajectory, state, xs=None, length=trajectory_length)

    states=jax.numpy.reshape(updatedstates, (num_samples, trajectory_length, environment.observation_size))
    actions=jax.numpy.reshape(updatedactions, (num_samples, trajectory_length, environment.action_size))
    return states, actions




#@jax.jit
#Input: 
#Envstep:  Enviroment.step : Callable
#Envreset: Enviroment.reset: Callable
#actions:  Actions: jax.Array 
#prng_key: Idk
#alpha_a: factor

#functionalized :)
def fo_update_action_sequence(environment, actions, prng_key, alpha_a):

    def total_reward(environment, actions, prng_key):
        
        def reward_step(states, action):
            return environment.step(states, action), states.reward

        initial_states = environment.reset(prng_key)
        _, rewards = jax.lax.scan(f=reward_step, init=initial_states, xs=actions)
        return jnp.sum(rewards, axis=1)

    grad = jax.grad(fun=total_reward, argnums=1)(environment, actions, prng_key)
    improved_action_sequence = actions + alpha_a * grad
    return improved_action_sequence

def update_policy(states, actions, train_state):

    params = train_state.policy_params
    optimizer_state = train_state.optimizer_state
    optimizer = train_state.optimizer

    @jax.jit
    def loss_fn_policy(params, state, targets):
        policy_outputs = train_state.policy_model.apply(params, state)
        return jnp.mean(0.5*jnp.square(policy_outputs - targets))

    for state, action in zip(states, actions):
        grads = jax.grad(loss_fn_policy, argnums=0)(params, state, action)
        updates, new_opt_state = optimizer.update(grads, optimizer_state)
        new_params = optax.apply_updates(train_state.policy_params, updates)
        
        train_state = train_state.replace(optimizer_state=new_opt_state)
        train_state = train_state.replace(policy_params=new_params)

    return train_state

def train(
    environment,
    trajectory_length: int,
    num_samples: int,
    epochs: int,
    alpha_a: float,
    progress_fn=None):

    # get a random key
    key = jax.random.PRNGKey(0)
    new_key, subkey = jax.random.split(key)

    # Define the policy and initialize it
    observation_size = int(environment.observation_size)
    action_size = int(environment.action_size)
    policy_model = DeterministicPolicy(observation_size=observation_size,action_size= action_size)
    policy_params = policy_model.init(key, jnp.ones((observation_size,)))
    
    # Define the optimizer
    optimizer = optax.adam(learning_rate=1e-3)
    optimizer_state = optimizer.init(policy_params)

    # Initialize the training state
    train_state = TrainState(
        policy_model=policy_model, 
        policy_params=policy_params,
        optimizer_state=optimizer_state,
        optimizer=optimizer)
    
    # Wrap the environment to allow vmapping
    environment = envs.training.wrap(environment, episode_length=trajectory_length,)
    
    
    # 1. run m episodes of the environment using the policy, of length trajectory_length
    # 2. collect the states and actions encountered in each episode
    # 3. for each episode initialize an array which has the sequence of actions taken by the policy
    # 4. rerun the environment, using the array of actions as input, and calculate the total reward
    # 5. calculate the gradient of the total reward with respect to the array of actions
    # 6. update the array of actions
    # 7. perform supervised learning on the policy using the array of actions
    x_data,y_data = [],[]


    for _ in range(epochs):
        # 1 - 3
        
        # update rng keys
        key1, key2 = jax.random.split(new_key)
        new_key = key1
        subkeys = jax.random.split(key2, num_samples)

        print(f"Num samples hausif: {num_samples}")
        trajectories = generate_trajectory_parallel(environment, train_state, trajectory_length, num_samples, subkeys)
        print(trajectories)
        
        
        
        # 4 - 6, first order update
        
        
            
        # 7
        for j in range(epochs):
            for states, actions in updated_trajectories.values():
                #print("states:",states,"actions:",actions,"train_state:",train_state)
                train_state = update_policy(states, actions, train_state)
            print("sample:",j,"out of ",epochs)
            
    return functools.partial(train_state.policy_model.apply, variables=train_state.policy_params)

            
