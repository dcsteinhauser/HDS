
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
from functools import partial

from ...dyn_model.Predict import make_inference_fn
from ...dyn_model.TuneModel import TuneModel
# from original.Pendulum import State as OriginalState


@struct.dataclass
class State(base.Base):
  """Environment state for training and inference."""

  state: jax.Array
  obs: jax.Array
  reward: jax.Array
  done: jax.Array
  metrics: Dict[str, jax.Array] = struct.field(default_factory=dict)
  info: Dict[str, Any] = struct.field(default_factory=dict)
  

class LearnedPendulum(PipelineEnv):
  #didnt change from default
  def __init__(self,action_size,observation_size, backend='generalized', **kwargs):
    path = 'src/envs/learnedenv/inverted_pendulum.xml'
    sys = mjcf.load(path)

    n_frames = 2

    # self.target = target
    if backend in ['spring', 'positional']:
      sys = sys.replace(dt=0.005)
      n_frames = 4
    
    self.inference_func = make_inference_fn(observation_size,action_size)
    kwargs['n_frames'] = kwargs.get('n_frames', n_frames)
    super().__init__(sys=sys, backend=backend, **kwargs)
  
  #@partial(jax.jit, static_argnums=(0,))
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
    qqd = jp.concatenate([pipeline_state.q, pipeline_state.qd])
    
    obs = self._get_obs(pipeline_state)
    reward, done = jp.zeros(2)
    metrics = {}
  
    return State(state=qqd,obs=obs,reward=reward,done=done,metrics=metrics, info={"init_qqd": qqd, "init_obs": obs})
  

  #@partial(jax.jit, static_argnums=(0))
  def step(self, state: State, action: jax.Array, params: dict) -> State:
    """Run one timestep of the environment's dynamics."""
    #current and next statex
    qqd= state.state
    qqd_next = self.inference_func(jp.concatenate((qqd,action)),params) # add model inference here
    obs_prev = qqd
    obs_next = qqd_next

    x_pos = obs_next[0]
    pseudo_angle = jp.cos(obs_next[1])
    x_vel = obs_next[2]
    angle_vel = obs_next[3]
    

    done = jax.lax.cond(jp.logical_and(jp.logical_and(jp.square(x_pos) < 0.001, jp.square(pseudo_angle) < 0.001), 
                                       jp.logical_and(jp.square(angle_vel) < 0.001, jp.square(x_vel) < 0.001)), lambda x: 1.0, lambda x: 0.0, None)
    reward = jax.lax.cond(done, lambda x: jp.square(action).sum(), lambda x: -1*(pseudo_angle)**2 - 1*angle_vel**2 - 2*x_pos**2 - 0.5*x_vel**2, None)

    return jax.lax.cond(done, lambda x: State(state.info["init_qqd"], state.info["init_obs"], jp.array(0.0), 0.0, metrics={}, info=state.info), 
                        lambda x: State(qqd_next, obs_next, reward, done, metrics={}, info=state.info), None)


  @property
  def action_size(self):
    return 1

  def _get_obs(self, pipeline_state: base.State) -> jax.Array:
    """Observe cartpole body position and velocities."""
    return jp.concatenate([pipeline_state.q, pipeline_state.qd])

  @partial(jax.jit, static_argnums=(0,))
  def tunemodel(self, params, obs_sequence,action_sequence):
    obs_t = obs_sequence[:-1]
    obs_tp = obs_sequence[1:]
    action_t = action_sequence[:-1]
    Y_data = obs_tp

    return TuneModel(obs_t,action_t,Y_data,params,self.inference_func)
  