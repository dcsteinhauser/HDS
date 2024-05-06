
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
  


class InvertedPendulum(PipelineEnv):
  #didnt change from default
  def __init__(self, backend='positional',target = jp.array([0]), **kwargs):
    path = 'src/env/inverted_pendulum.xml'
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
    # configuration if pendulum is just supposed to go to downward position not necessarily origin
    # def f1():
    #   distancex = jp.array(0, float)
    #   distancey = jp.array(1+jp.cos(obs[1]), float)
    #   wx,wp = 0,0
    #   return distancex,distancey,wx,wp
    # # config if downward at origin 
    # def f2():
    #   distancex = jp.array(obs[0], float)
    #   distancey = jp.array(1-jp.cos(obs[1]), float)
    #   wx,wp = 10,2
    #   return distancex,distancey,wx,wp

    #determining target randomly unless otherwise given
    # if self.target is not None:
    #   target = self.target
    # else:
    #   target = jax.random.bits(rng3, shape=(1,))%2

    #if statement, depending pn what the target says
    # distancex,distancey,wx,wp = jax.lax.cond(target[0] == 0, f1, f2)
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
    reward = jax.lax.cond(done, lambda x: jp.square(action).sum(), lambda x: -1*(pseudo_angle)**2 - 1*angle_vel**2 - 2*x_pos**2 - 0.5*x_vel**2, None)

    return jax.lax.cond(done, lambda x: State(pipeline_state, obs_prev, reward, done, metrics={}), 
                        lambda x: State(pipeline_state_next, obs_next, reward, done, metrics={}), None)

    #target
    # target= state.target
    #reward function weights
    # wa,wvel,wang,wp,wx = 1,1,1,state.wp,state.wx
    
    #function in case the sequence is done, sets the next state to current state and applies heavily negative reward
    # def step1(pipeline_state,pipeline_state_next,obs_prev,obs_next,done,wa,wvel,wang,wp,wx):
    #   reward = -100000.0
    #   reward = jp.array(reward, float)
    #   ps1 = pipeline_state
    #   obs = obs_prev
    #   done=done
    #   #print(reward)
    #   return ps1,obs,reward,done
    
    # #normal situation, with check if next state results in done flag
    # def step2(pipeline_state,pipeline_state_next,obs_prev,obs_next,done,wa,wvel,wang,wp,wx):

    #   def rew1(reward):
    #     return reward
      
    #   def rew2(reward):
    #     return -100000.0

    #   ps1 = pipeline_state_next
    #   obs = obs_next
    #   reward = -wx*(obs_next[0])**4
    #   done = jp.where(jp.abs(obs[0]) > 9.0, 1.0, 0.0)
    #   reward=jax.lax.cond(state.done==1,rew2,rew1,reward)
    #   #print(reward)

    #   return ps1,obs,reward,done
    
    # done= state.done
    # #update function, if condition basically based on if its done or not
    # pipeline_state_next,obs,reward,done = jax.lax.cond(state.done == 1, step1, step2, pipeline_state,pipeline_state_next,obs_prev,obs_next,state.done,wa,wvel,wang,wp,wx)    
    # reward = jp.array(reward, float)
    
    
    # return state.replace(
    #     pipeline_state=pipeline_state_next, obs=obs, reward=reward, done=done
    # )


#rest here is default
  @property
  def action_size(self):
    return 1

  def _get_obs(self, pipeline_state: base.State) -> jax.Array:
    """Observe cartpole body position and velocities."""
    return jp.concatenate([pipeline_state.q, pipeline_state.qd])
