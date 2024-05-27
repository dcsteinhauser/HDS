import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn

class StochasticPolicy(nn.Module):

    observation_size: int
    action_size: int

    def setup(self):
        #self.dense1 = nn.Dense(32)
        #self.dense2 = nn.Dense(64)
        self.dense3 = nn.Dense(128)
        #self.dense4 = nn.Dense(32)
        #self.dense5 = nn.Dense(10)
        self.dense6 = nn.Dense(self.action_size)


    def __call__(self, x, noise=1, rng_key=None):
        #x = self.dense1(x)
        #x = jax.nn.relu(x)
        #x = self.dense2(x)
        #x = jax.nn.relu(x)
        x = self.dense3(x)
        x = jax.nn.relu(x)
        #x = self.dense4(x)
        #x = jax.nn.relu(x)
        #x = self.dense5(x)
        #x = jax.nn.relu(x)
        x = self.dense6(x)
        x = jax.nn.tanh(x)

        if rng_key is None:
            return x

        x = x + jax.nn.tanh(jax.random.normal(rng_key, x.shape)) * noise

        return x

