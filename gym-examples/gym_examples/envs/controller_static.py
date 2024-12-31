from controller_template import ControllerTemplate

class StaticController(ControllerTemplate):

    def __call__(self, observation):
        return (0,0)
