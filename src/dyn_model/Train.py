from DynamicsModel import model as DynamicsModel
import jax.numpy as jnp
import optax
import jax
from flax.training import train_state



def mse_loss(params, apply_fn, x, y):
    preds = apply_fn({'params': params}, x)
    return jnp.mean((preds - y) ** 2)


def create_train_state(rng, learning_rate, observation_size,action_size):
    model = DynamicsModel(output_size=observation_size,input_size=observation_size+action_size)
    params = model.init(rng, jnp.ones([1, observation_size+action_size]))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def train_step(state, batch):
    def loss_fn(params):
        loss = mse_loss(params, state.apply_fn, batch['x'], batch['y'])
        return loss

    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state


def train(rng, X_train, y_train, X_test, y_test, observation_size,action_size, learning_rate=0.001, num_epochs=100, batch_size=32):
    state = create_train_state(rng, learning_rate, observation_size,action_size)
    num_batches = X_train.shape[0] // batch_size
    
    for epoch in range(num_epochs):
        perm = jax.random.permutation(rng, X_train.shape[0])
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]

        for i in range(num_batches):
            batch_idx = perm[i * batch_size:(i + 1) * batch_size]
            batch = {'x': X_train_shuffled[batch_idx], 'y': y_train_shuffled[batch_idx]}
            state = train_step(state, batch)

        if epoch % 10 == 0:
            train_loss = mse_loss(state.params, state.apply_fn, X_train, y_train)
            test_loss = mse_loss(state.params, state.apply_fn, X_test, y_test)
            print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    return state.params