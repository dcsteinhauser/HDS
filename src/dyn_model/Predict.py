from .DynamicsModel import model as DynamicsModel
import jax.numpy as jnp
import orbax.checkpoint as ocp
import os 

def make_inference_fn(observation_size, action_size):
    def inference_fun(input,params):
        model = DynamicsModel(output_size=observation_size,input_size=observation_size+action_size)
        return model.apply({'params':params},input)
    return inference_fun

def pretrained_params():
    ckpt_dir = os.path.abspath('src/dyn_model/params/1')
    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    params = ckptr.restore(ckpt_dir)
    return params