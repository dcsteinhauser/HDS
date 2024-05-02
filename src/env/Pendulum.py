
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

@struct.dataclass
class State(base.Base):
  """Environment state for training and inference."""

  pipeline_state: Optional[base.State]
  obs: jax.Array
  reward: jax.Array
  done: jax.Array
  target: jax.Array
  distancex: jax.Array
  distancey: jax.Array
  wx: jax.Array
  wp: jax.Array
  metrics: Dict[str, jax.Array] = struct.field(default_factory=dict)
  info: Dict[str, Any] = struct.field(default_factory=dict)
  


class InvertedPendulum(PipelineEnv):

  def __init__(self, backend='generalized',target = jp.array([0]), **kwargs):
    path = 'src/env/inverted_pendulum.xml'
    sys = mjcf.load(path)

    n_frames = 2

    self.target = target
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
        rng1, (self.sys.q_size(),), minval=-0.01, maxval=0.01
    )
    qd = jax.random.uniform(
        rng2, (self.sys.qd_size(),), minval=-0.01, maxval=0.01
    )
    pipeline_state = self.pipeline_init(q, qd)
    obs = self._get_obs(pipeline_state)

    def f1():
      distancex = jp.array(0, float)
      distancey = jp.array(1+jp.cos(obs[1]), float)
      wx,wp = 0,0
      return distancex,distancey,wx,wp

    def f2():
      distancex = jp.array(obs[0], float)
      distancey = jp.array(1-jp.cos(obs[1]), float)
      wx,wp = 100,2
      return distancex,distancey,wx,wp


    if self.target is not None:
      target = self.target
    else:
      target = jax.random.bits(rng3, shape=(1,))%2

    
    distancex,distancey,wx,wp = jax.lax.cond(target[0] == 0, f1, f2)
    reward, done = jp.zeros(2)
    metrics = {}

    return State(pipeline_state, obs, reward, done,target,distancex,distancey, wx,wp, metrics)
  
  @partial(jax.jit, static_argnums=(0,))
  def step(self, state: State, action: jax.Array) -> State:
    """Run one timestep of the environment's dynamics."""
    
    
    pipeline_state = self.pipeline_step(state.pipeline_state, action)
    obs = self._get_obs(pipeline_state)
    target= state.target
    wa,wvel,wang=10,100,1
    wp,wx = state.wp,state.wx

    #reward = -wx*(state.distancex **2 + state.distancey **2) -wp*(jp.abs(obs[1])-jp.pi)**2 - wa*(action)**2 - wvel*(obs[2]**2)  -wang*(obs[3]**2)
    reward = -wx*(obs[0])**2 - wa *(action)**2 -wx*(obs[1])**2
    reward =jp.array(reward[0],float)
    
    def negativerew(reward):
      reward = -1000.0
      return reward
    
    def rew(reward):
      return reward
    
    reward=jax.lax.cond(state.done==1,negativerew,rew,reward)
    done = jp.where(jp.abs(obs[0]) > 1.0, 1.0, 0.0)

    return state.replace(
        pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
    )

  @property
  def action_size(self):
    return 1

  def _get_obs(self, pipeline_state: base.State) -> jax.Array:
    """Observe cartpole body position and velocities."""
    return jp.concatenate([pipeline_state.q, pipeline_state.qd])
