#controller_ORCA.py

import rvo2 
from controller_template import ControllerTemplate

import numpy as np 
from typing import Tuple, List, Dict


class ORCA_controller(ControllerTemplate):
    '''Every UAV that uses ORCA will get its own local RVO2 simulator 
    which will calculate necessary attribute - new velocity.'''

    def __init__(self, max_acceleration, max_heading_change):
        



        pass 




    def __call__(self, observation):
        # take a step in RVO2
        # collect and send new position to dynamics 
        # uav.dynamics will not perform any calculations related to accleration and instantaneous heading change using these values 
        # uav.dynamics will perform some calculation to convert velocity to speed, direction, new position and new current heading
        # uav.dynamics will send values directly to UAV to update position and heading
        pass 
