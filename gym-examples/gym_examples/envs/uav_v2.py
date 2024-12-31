
from uav_v2_template import UAV_v2_template

class UAV_v2(UAV_v2_template):

    def __init__(self, controller, dynamics,sensor,radius):
        super().__init__(controller, dynamics,sensor,radius)

    def assign_start_end(self, start, end):
        return super().assign_start_end(start, end)
    

    def get_action(self, observation):
        action = self.controller(observation)
        return action

    def step(self,action):
        self.dynamics.update(action)




