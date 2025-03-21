from typing import Tuple, List, Dict
import numpy as np

from controller_template import ControllerTemplate

import pyorca as ORCA

class ORCA_controller(ControllerTemplate):
    
    def __init__(self, max_acceleration, max_heading_change, tau, dt):
        super().__init__(max_acceleration, max_heading_change)
        self.tau = tau
        self.dt = dt

    
    
    
    
    
    def __call__(self, observation:List[Dict]) -> Tuple: #action
       '''
       input: observation list of dicts, each dict consists of agent information. 
       The first dict consists of self information. 

       Return: action tuple. 
       '''
       # creating Agent according to pyORCA's implementation
       SPEED = 15 
       _agent = observation[0]
       _agent_position = _agent['current_position'].x, _agent['current_position'].y
       _agent_speed = SPEED #_agent['current_speed']
       _agent_heading = _agent['current_heading']
       _agent_velocity = [_agent_speed*np.cos(_agent_heading), 
                          _agent_speed*np.sin(_agent_heading)]
       
       ref_final_point = _agent['current_position'] - _agent['end'] #Point obj
       x = np.array([ref_final_point.x, ref_final_point.y])
       pref_vel = ORCA.normalized(-x) * _agent_speed
       
       print(f'speed: {_agent_speed}, heading: {_agent_heading}')


       agent = ORCA.Agent(position=_agent_position, 
                          velocity=_agent_velocity,
                          radius=_agent['radius'],
                          max_speed=_agent['max_speed'],
                          pref_velocity= pref_vel)
       
       candidates = []
       
       for other_agent_info in observation[1:]:
          _other_agent = other_agent_info
          _other_agent_speed = SPEED # _other_agent['current_speed']
          _other_agent_heading = _other_agent['current_heading']
          _other_agent_velocity = [_other_agent_speed*np.cos(_other_agent_heading)
                                  ,_other_agent_speed*np.sin(_other_agent_heading)]
          ref_final_point = _other_agent['current_position'] - _other_agent['end'] #Point obj
          x = np.array([ref_final_point.x, ref_final_point.y])
          pref_vel = ORCA.normalized(-x) * _other_agent_speed

          other_agent = ORCA.Agent(position =_other_agent['current_position'], 
                          velocity = _other_agent_velocity,
                          radius = _other_agent['radius'],
                          max_speed = _other_agent['max_speed'],
                          pref_velocity = pref_vel)
          
          candidates.append(other_agent)
        
       new_vel, line = ORCA.orca(agent, candidates, self.tau, self.dt)
       print(f'new_vel: {new_vel}') 
       return new_vel
       
        
       pass 



