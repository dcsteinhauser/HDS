
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
  def __init__(self, backend='generalized', **kwargs):
    path = 'src/env/inverted_pendulum.xml'
    sys = mjcf.load(path)

    n_frames = 2

    if backend in ['spring', 'positional']:
      sys = sys.tree_replace({'opt.timestep': 0.005})
      n_frames = 4

    kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

    super().__init__(sys=sys, backend=backend, **kwargs)


  def reset(self, rng: jax.Array) -> State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jax.random.split(rng, 3)

    q = self.sys.init_q 
    #initialize the entries of q separately
    q = q.at[0].add(jax.random.uniform(key=rng1, minval=-6, maxval=6))
    q = q.at[1].add(jax.random.uniform(key=rng1, minval=-0.05, maxval=0.05))

    #qd = jax.random.uniform(rng2, (self.sys.qd_size(),), minval=-3, maxval=3)
    qd = jax.numpy.array([0.0,0.0])
    qd.at[0].set(jax.random.uniform(key=rng2, minval=-0.01, maxval=0.01))
    qd.at[1].set(jax.random.uniform(key=rng2, minval=-0.01, maxval=0.01))
    pipeline_state = self.pipeline_init(q, qd)
    obs = self._get_obs(pipeline_state)
    reward, done = jp.zeros(2)
    metrics = {}

    return State(pipeline_state, obs, reward, done, metrics)


  

  def step(self, state: State, action: jax.Array) -> State:
    """Run one timestep of the environment's dynamics."""
    #current and next state
    action_min = self.sys.actuator.ctrl_range[:, 0]
    action_max = self.sys.actuator.ctrl_range[:, 1]
    action = (action + 1) * (action_max - action_min) * 0.5 + action_min

    pipeline_state = self.pipeline_step(state.pipeline_state, action)
    #current and next observations
    obs = self._get_obs(pipeline_state)


    x_pos = obs[0]
    pseudo_angle = jp.cos(obs[1])

    x_vel = obs[2]
    angle_vel = obs[3]

    posdone = jp.logical_and(jp.square(x_pos) < 0.00000001, jp.square(x_vel) < 0.00000001)
    
    #done flag
    done = jp.where(posdone, 1.0, 0.0)
    bool_done = done>0.5
    #boundarycond = jax.lax.cond(x_pos>9.8, lambda x: 100.0, lambda x: 0.0, None)
    reward=jax.lax.cond(bool_done, lambda x: 0.0, lambda x: -(200*action[0]**2 - 0*(pseudo_angle)+ 0*angle_vel**2  + 20*(x_pos)**2 + 1.1*x_vel**2), None)
    #if done is True, reset 
    return state.replace(
        pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
    )

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
