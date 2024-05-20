from .DynamicsModel import model as DynamicsModel
import jax.numpy as jnp
import orbax.checkpoint as ocp


def make_inference_fn(observation_size, action_size):
    def inference_fun(input,params):
        model = DynamicsModel(output_size=observation_size,input_size=observation_size+action_size)
        return model.apply({'params':params},input)
    return inference_fun