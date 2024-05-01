import jax
from flax import linen as nn

class DeterministicPolicy(nn.Module):

    observation_size: int
    action_size: int

    def setup(self):
        self.dense1 = nn.Dense(5)
        self.dense2 = nn.Dense(2)
        self.dense3 = nn.Dense(self.action_size)

    def __call__(self, x):
        x = self.dense1(x)
        x = jax.nn.tanh(x)
        x = self.dense2(x)
        x = jax.nn.tanh(x)
        x = self.dense3(x)
        return x

Batch_DeterministicPolicy = nn.vmap(DeterministicPolicy,in_axes = 0,out_axes=0, variable_axes={'params': None},
    split_rngs={'params': False})