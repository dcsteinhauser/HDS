import functools

from brax.training.types import Params
from brax.training.types import PRNGKey
import flax
import jax
import jax.numpy as jnp
import optax
from functools import partial

from src.policy.StochasticPolicy import StochasticPolicy
from src.envs.learnedenv.LearnedPendulum import LearnedPendulum
from src.envs.train.Pendulum import State
from src.dyn_model.Predict import pretrained_params


@flax.struct.dataclass
class TrainState:
    policy_params: Params
    optimizer_state: optax.OptState
    exploration_noise: float
    exploration_noise_decay: float
    dynamics_model_params: Params


def make_policy(network, params):

    def policy(obs):
        return network.apply(params, obs)

    return policy


def train(
    env,
    trajectory_length: int,
    num_samples: int,
    epochs: int,
    inner_epochs: int,
    alpha_a: float,
    aggregation_factor_beta: float,
    init_learning_rate: float,
    init_noise=1.0,
    noise_decay=0.99,
    progress_fn=None,
):
    # initialize environments
    k_NON_BATCHED_ENV = env
    k_LEARNED_ENV = LearnedPendulum(
        action_size=int(env.action_size), observation_size=int(env.observation_size)
    )
    dynamics_pretrained_params = pretrained_params()

    # initialize array for whether to use learned or original env
    num_from_learned_env = int(num_samples * aggregation_factor_beta)
    from_learned_env = jnp.ones((num_from_learned_env,))
    from_original_env = jnp.zeros((num_samples - num_from_learned_env,))
    use_learned_env = jnp.concatenate((from_learned_env, from_original_env))

    # for progress fn
    x_data = []
    y_data = []

    # get a random key
    key = jax.random.PRNGKey(0)
    new_key, subkey = jax.random.split(key)

    # Define the policy and initialize it
    observation_size = int(env.observation_size)
    action_size = int(env.action_size)
    k_POLICY_MODEL = StochasticPolicy(
        observation_size=observation_size, action_size=action_size
    )
    
    policy_params = k_POLICY_MODEL.init(
        key, jnp.ones((observation_size,))
    )

    # Define the optimizer
    scheduler = optax.exponential_decay(
        init_value=init_learning_rate, transition_steps=1000, decay_rate=0.99
    )

    # Combining gradient transforms using `optax.chain`.
    k_OPTIMIZER = optax.chain(
        optax.scale_by_adam(),  # Use the updates from adam.
        optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
        optax.scale(-1.0),
    )

    optimizer_state = k_OPTIMIZER.init(policy_params)

    # Initialize the training state
    train_state = TrainState(
        policy_params=policy_params,
        optimizer_state=optimizer_state,
        exploration_noise=init_noise,
        exploration_noise_decay=noise_decay,
        dynamics_model_params=dynamics_pretrained_params,
    )

    @partial(jax.vmap, in_axes=(None, None, 0, 0), axis_name="batch")
    def generate_trajectory_parallel(
        train_state: TrainState,
        trajectory_length: int,
        prng_keys: PRNGKey,
        use_learned_env: jnp.ndarray,
    ):

        def step_trajectory(state_carry, rng_key):
            action = k_POLICY_MODEL.apply(
                train_state.policy_params,
                state_carry.obs,
                train_state.exploration_noise,
                rng_key,
            )
            next_state = k_NON_BATCHED_ENV.step(state_carry, action)
            return next_state, (state_carry.obs, action, next_state.reward)

        def step_trajectory_learned_env(state_carry, rng_key):
            action = k_POLICY_MODEL.apply(
                train_state.policy_params,
                state_carry.obs,
                train_state.exploration_noise,
                rng_key,
            )
            next_state = k_LEARNED_ENV.step(
                state_carry, action, params=train_state.dynamics_model_params
            )
            return next_state, (state_carry.obs, action, next_state.reward)

        def select_fn(input):
            _, (states, actions, rewards_future) = input
            return states, actions, rewards_future

        keys = jax.random.split(prng_keys, trajectory_length)

        states, actions, rewards_future = jax.lax.cond(
            use_learned_env,
            lambda _: select_fn(
                jax.lax.scan(
                    step_trajectory_learned_env, k_LEARNED_ENV.reset(prng_keys), xs=keys
                )
            ),
            lambda _: select_fn(
                jax.lax.scan(
                    step_trajectory, k_NON_BATCHED_ENV.reset(prng_keys), xs=keys
                )
            ),
            0,
        )

        states = jax.numpy.reshape(
            states, (trajectory_length, k_NON_BATCHED_ENV.observation_size)
        )
        actions = jax.numpy.reshape(
            actions, (trajectory_length, k_NON_BATCHED_ENV.action_size)
        )

        total_reward = jnp.sum(rewards_future)

        return states, actions, total_reward

    @partial(jax.vmap, in_axes=(0, None, 0, None, 0), out_axes=0, axis_name="batch")
    @jax.jit
    def fo_update_action_sequence(
        actions, train_state, prng_key, alpha_a, use_learned_env
    ):

        def total_reward(actions, prng_key):

            def reward_step(states, action):
                return k_NON_BATCHED_ENV.step(states, action), states.reward

            initial_states = k_NON_BATCHED_ENV.reset(prng_key)
            _, rewards = jax.lax.scan(f=reward_step, init=initial_states, xs=actions)
            return jnp.sum(rewards, axis=0)

        def total_reward_learned(actions, prng_key):

            def reward_step(states, action):
                return (
                    k_LEARNED_ENV.step(
                        states, action, params=train_state.dynamics_model_params
                    ),
                    states.reward,
                )

            initial_states = k_LEARNED_ENV.reset(prng_key)
            _, rewards = jax.lax.scan(f=reward_step, init=initial_states, xs=actions)
            return jnp.sum(rewards, axis=0)

        grad = jax.lax.cond(
            use_learned_env,
            lambda _: jax.grad(total_reward_learned, argnums=0)(actions, prng_key),
            lambda _: jax.grad(total_reward, argnums=0)(actions, prng_key),
            0,
        )
        new_actions = actions + alpha_a * grad
        return new_actions

    @jax.jit
    def update_policy(states, actions, train_state):
        params = train_state.policy_params
        optimizer_state = train_state.optimizer_state

        def loss_fn(params, states, actions):
            model_output = k_POLICY_MODEL.apply(params, states)
            return 0.5 * optax.losses.squared_error(model_output, actions).mean()

        value, grad = jax.value_and_grad(loss_fn)(params, states, actions)
        updates, optimizer_state = k_OPTIMIZER.update(grad, optimizer_state)
        new_params = optax.apply_updates(params, updates)
        new_train_state = TrainState(
            policy_params=new_params,
            optimizer_state=optimizer_state,
            exploration_noise=train_state.exploration_noise,
            exploration_noise_decay=train_state.exploration_noise_decay,
            dynamics_model_params=train_state.dynamics_model_params,
        )
        return value, new_train_state

    for i in range(epochs):
        # update rng keys
        key1, key2, shuffle_key = jax.random.split(new_key, num=3)
        new_key = key1
        subkeys = jax.random.split(key2, num_samples)

        # shuffle use_learned_env
        shuffled_use_learned_env = jax.random.permutation(shuffle_key, use_learned_env)

        # generate trajectories
        trajectories = generate_trajectory_parallel(
            train_state, trajectory_length, subkeys, shuffled_use_learned_env
        )
        average_reward = trajectories[2]
        trajectories = trajectories[:2]

        # output progress
        progress_fn(x_data, y_data, i, jnp.mean(average_reward))

        # update action sequence
        states, new_actions = trajectories[0], fo_update_action_sequence(
            trajectories[1], train_state, subkeys, alpha_a, shuffled_use_learned_env
        )

        # supervised learning with early stopping
        for j in range(inner_epochs):
            for state_sequence, action_sequence in zip(states, new_actions):
                value, train_state = update_policy(
                    state_sequence, action_sequence, train_state
                )
            print("big epoch:", i, "small epoch:", j, "Loss", value)
            if value < 1e-5:
                break

        # update exploration noise
        train_state = train_state.replace(
            exploration_noise=train_state.exploration_noise
            * train_state.exploration_noise_decay
        )

    return functools.partial(
        make_policy, params=train_state.policy_params, network=k_POLICY_MODEL
    )
