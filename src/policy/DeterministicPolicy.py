import jax
from flax import linen as nn

class DeterministicPolicy(nn.Module):

    observation_size: int
    action_size: int

    def setup(self):
        self.dense1 = nn.Dense(self.observation_size //2)
        self.dense2 = nn.Dense(self.action_size)

    def __call__(self, x):
        x = self.dense1(x)
        x = jax.nn.tanh(x)
        x = self.dense2(x)
        return x
