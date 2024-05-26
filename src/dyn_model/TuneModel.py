from .DynamicsModel import DynamicsModel
import jax.numpy as jnp
import optax
import jax
from flax.training import train_state
from functools import partial
 
#@jax.vmap
def mse_loss(params,inputs,inference_fn,y_data):
    preds = inference_fn(inputs,params)
    return jnp.mean((preds - y_data) ** 2)


def create_train_state(params,learning_rate,inference_fn):
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=inference_fn, params=params, tx=tx)


#Xdata,ydata,params_inference_fun
#@partial(jax.vmap, in_axes=(0,0,0,None,None))
def TuneModel (obs_t,acts_t,y_data,params,inference_fn, learning_rate=0.001, num_epochs=100, batch_size=32):
    TrainState = create_train_state(params,learning_rate,inference_fn)
    inputs = jnp.concatenate((obs_t, acts_t), axis=1)
    _,grad = jax.value_and_grad(mse_loss)(params,inputs,inference_fn,y_data)
    TrainState = TrainState.apply_gradients(grads=grad)
    params = TrainState.params
    return params
