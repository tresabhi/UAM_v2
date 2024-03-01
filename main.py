import matplotlib.pyplot as plt
from simulator import Simulator
from matplotlib.animation import FuncAnimation
import geopandas as gpd
from utils import static_plot


 #TODO - Complete the following checks 
'''
Once the main simulator.py is built, it should have the following 
1) make sure there is at least 2 vertiports at all time 
2) There is at least one uav - if 0 UAV, no need to run simulation(simulation should not run, but check to make sure)
3) make sure location name is valid
4) make sure vertiports are not on top of buildings and other structures 

'''




if __name__ == '__main__':
    
    sim = Simulator('Austin, Texas, USA', 7, 1, sleep_time=0.05, total_timestep = 1500)
    #*Plotting Logic
    # #TODO - Use FuncAnimation to animate the path of the UAV
    # #TODO - call a plotter function here that encapsulates this loop 

    plt.ion() 
    fig, ax = plt.subplots()
    static_plot(sim, ax, gpd)
    sim.RUN_SIMULATOR(fig, ax, static_plot,sim, gpd)