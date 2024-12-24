import matplotlib.pyplot as plt
from simulator_basic import SimulatorBasic


#! - Complete the following checks


# TODO - 3) make sure location name is valid
# TODO - 4) make sure vertiports are not on top of buildings and other structures
# TODO - 5) run headless(without video)
# TODO - 6) be able to dynamically add locations using strings


if __name__ == "__main__":

    # #controller = ZeroController()
    # controller = CollisionController()
    # controller_predict = controller.get_action

    sim = SimulatorBasic(
        "Austin, Texas, USA", 8, 5, sleep_time=0.01, total_timestep=1500, airspace_tag_list=[("building", "hospital"),("aeroway", "aerodrome")]
    )

    # fig, ax initialization
    plt.ion()
    fig, ax = sim.render_init()

    # * remember this is a convinience function
    sim.run_simulator(fig, ax)
    sim.reset()
    #! Need to clear the current axes and plot only the vertiports and uavs

    sim.run_simulator(fig, ax)

    # *Plotting Logic
    # #TODO - Use FuncAnimation to animate the path of the UAV
    # #TODO - call a plotter function here that encapsulates this loop

    # logfile = '</change /to /logfile /path>'
    # print('Simulation initialization -  ')
    # print('Current working dir', os.getcwd())
    # os.chdir(path='gym-examples/gym_examples/envs/assets')
    # print('Current working dir: ', os.getcwd())
    # print('state log files will be saved in ', logfile)