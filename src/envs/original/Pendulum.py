
from brax import base
from brax.envs.base import PipelineEnv
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp

import abc
from typing import Any, Dict, List, Optional, Sequence, Union
from brax import base
from brax.generalized import pipeline as g_pipeline
from brax.io import image
from brax.mjx import pipeline as m_pipeline
from brax.positional import pipeline as p_pipeline
from brax.spring import pipeline as s_pipeline
from flax import struct
import jax
import mujoco
from mujoco import mjx
import numpy as np
from functools import partial
from jax import vmap

# Ugliest code I have seen in a while - Daniel
@struct.dataclass
class State(base.Base):
  """Environment state for training and inference."""

  pipeline_state: Optional[base.State]
  obs: jax.Array
  reward: jax.Array
  done: jax.Array
  # supposed target value to get your pendulum to go
  # target: jax.Array
  # distancex: jax.Array
  # distancey: jax.Array
  # #weights for target difference and position (like the apper)
  # wx: jax.Array
  # wp: jax.Array

  metrics: Dict[str, jax.Array] = struct.field(default_factory=dict)
  info: Dict[str, Any] = struct.field(default_factory=dict)
  


class Pendulum(PipelineEnv):
  #didnt change from default
  def __init__(self, backend='generalized', **kwargs):
    path = 'src/envs/original/inverted_pendulum.xml'
    sys = mjcf.load(path)

    n_frames = 2

    # self.target = target
    if backend in ['spring', 'positional']:
      sys = sys.replace(dt=0.005)
      n_frames = 4
    
    

    kwargs['n_frames'] = kwargs.get('n_frames', n_frames)
    super().__init__(sys=sys, backend=backend, **kwargs)
  
  @partial(jax.jit, static_argnums=(0,))
  def reset(self, rng: jax.Array) -> State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2, rng3 = jax.random.split(rng,4)

    q = self.sys.init_q + jax.random.uniform(
        rng1, (self.sys.q_size(),), minval=-5, maxval=5
    )
    qd = jax.random.uniform(
        rng2, (self.sys.qd_size(),), minval=-5, maxval=5
    )
    pipeline_state = self.pipeline_init(q, qd)
    obs = self._get_obs(pipeline_state)
    reward, done = jp.zeros(2)
    metrics = {}

    return State(pipeline_state, obs, reward, done, metrics)
  

  @partial(jax.jit, static_argnums=(0))
  def step(self, state: State, action: jax.Array) -> State:
    """Run one timestep of the environment's dynamics."""
    #current and next state
    pipeline_state=state.pipeline_state
    pipeline_state_next = self.pipeline_step(state.pipeline_state, action)
    #current and next observations
    obs_prev = self._get_obs(state.pipeline_state)
    obs_next = self._get_obs(pipeline_state_next)

    x_pos = obs_next[0]
    pseudo_angle = jp.cos(obs_next[1])
    x_vel = obs_next[2]
    angle_vel = obs_next[3]
    

    done = jax.lax.cond(jp.logical_and(jp.logical_and(jp.square(x_pos) < 0.001, jp.square(pseudo_angle) < 0.001), 
                                       jp.logical_and(jp.square(angle_vel) < 0.001, jp.square(x_vel) < 0.001)), lambda x: 1.0, lambda x: 0.0, None)
    reward = jax.lax.cond(done, lambda x: jp.square(action).sum(), lambda x: -1*(pseudo_angle)**2 - 1*angle_vel**2  - 1.5*(x_pos)**2 - 0.5*x_vel**2, None)

    return jax.lax.cond(done, lambda x: State(pipeline_state, obs_prev, reward, done, metrics={}), 
                        lambda x: State(pipeline_state_next, obs_next, reward, done, metrics={}), None)



#rest here is default
  @property
  def action_size(self):
    return 1

  def _get_obs(self, pipeline_state: base.State) -> jax.Array:
    """Observe cartpole body position and velocities."""
    return jp.concatenate([pipeline_state.q, pipeline_state.qd])
