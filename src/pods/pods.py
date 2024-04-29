import functools

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
def generate_trajectory(environment, train_state: TrainState, trajectory_length: int, prng_key: PRNGKey):
    
    states = jnp.zeros((trajectory_length, environment.observation_size))
    actions = jnp.zeros((trajectory_length, environment.action_size))
    total_reward = jnp.zeros(1)
    state: State = environment.reset(prng_key)
    
    for i in range(trajectory_length):
        action = train_state.policy_model.apply(train_state.policy_params, state.obs)
        next_state = environment.step(state, action)
        states=states.at[i].set(state.obs)
        total_reward += state.reward
        actions=actions.at[i].set(action)
        state = next_state

    return states, actions, total_reward


#@jax.jit
def fo_update_action_sequence(environment, states, actions, prng_key, alpha_a):

    # print(states)
    # print(actions)
    # print(prng_key)
    
    def total_reward(environment, states, actions):
        total_reward = jnp.zeros(1)
        state = environment.reset(prng_key)

        for i in range(states.shape[0]):
            action = actions[i]
            next_state = environment.step(state, action)
            total_reward += next_state.reward
            state = next_state
        return jnp.reshape(total_reward, ())
    
    grad = jax.grad(fun=total_reward, argnums=2)(environment, states, actions)
    #print(grad)
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

def progress_f(x_data,y_data,epoch,reward):
    x_data.append(epoch)
    y_data.append(reward)
    clear_output(wait=True)
    plt.xlabel('epoch')
    plt.ylabel('total reward')
    plt.plot(x_data, y_data)
    plt.show()


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
    
    if progress_fn is None:
        progress_fn = progress_f
    
    # 1. run m episodes of the environment using the policy, of length trajectory_length
    # 2. collect the states and actions encountered in each episode
    # 3. for each episode initialize an array which has the sequence of actions taken by the policy
    # 4. rerun the environment, using the array of actions as input, and calculate the total reward
    # 5. calculate the gradient of the total reward with respect to the array of actions
    # 6. update the array of actions
    # 7. perform supervised learning on the policy using the array of actions
    x_data,y_data = [],[]
    
    for epoch in range(epochs):
        trajectories = dict()
        # 1 - 3
        
        for i in range(num_samples):
            new_key2, subkey2 = jax.random.split(new_key)
            new_key = new_key2

            states, actions, total_reward = generate_trajectory(environment, train_state, trajectory_length, subkey2)
            trajectories[i] = (states, actions, subkey2)

            print("sample:",i,"out of ",num_samples,"total reward:",total_reward,"epoch:",epoch)
        
        progress_fn(x_data,y_data, epoch, total_reward)
        # 4 - 6, first order update
        updated_trajectories = jax.tree_util.tree_map(lambda states_actions_initialkey: 
                               (states_actions_initialkey[0], 
                                fo_update_action_sequence(environment, states_actions_initialkey[0], states_actions_initialkey[1], states_actions_initialkey[2], alpha_a)), 
                                trajectories, is_leaf=lambda x: isinstance(x, tuple) and len(x) == 3)
        
            
        # 7
        for j in range(epochs):
            for states, actions in updated_trajectories.values():
                #print("states:",states,"actions:",actions,"train_state:",train_state)
                train_state = update_policy(states, actions, train_state)
            print("sample:",j,"out of ",epochs)
            
    return functools.partial(train_state.policy_model.apply, variables=train_state.policy_params)

            
