from controller_template import ControllerTemplate


class ORCA_controller(ControllerTemplate):
    def __init__(self, max_acceleration, max_heading_change, tau, dt):
        super().__init__(max_acceleration, max_heading_change)
        self.tau = tau
        self.dt = dt

    def __call__(self, observation):
        agent = observation[0]
        candidates = observation[1:]
        new_vel, all_lines = self.orca(agent, candidates, self.tau, self.dt)

        action = new_vel
        
        return action
    
    def orca():
        pass

    


