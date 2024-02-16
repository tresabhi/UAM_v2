import airspace
import uav
from airtrafficcontroller import ATC
import numpy as np
import matplotlib.pyplot as plt
from simulator import Simulator
from matplotlib.animation import FuncAnimation
import geopandas as gpd
import plotter
import time


 #TODO - Complete the following checks 
'''
Once the main simulator.py is built, it should have the following 
1) make sure there is at least 2 vertiports at all time 
2) There is at least one uav - if 0 UAV, no need to run simulation(simulation should not run, but check to make sure)
3) make sure location name is valid
4) make sure vertiports are not on top of buildings and other structures 

'''

if __name__ == '__main__':
    
    sim = Simulator('Austin, Texas, USA', 11, 5)
    #*Plotting Logic
    # #TODO - Use FuncAnimation to animate the path of the UAV
    # #TODO - call a plotter function here that encapsulates this loop 

    plt.ion() 
    fig, ax = plt.subplots()
    # vertiports_gs = gpd.GeoSeries(sim.vertiports)
    
    def static_plot():
        sim.airspace.location_utm_gdf.plot(ax=ax, color='gray', linewidth=0.6)
        sim.airspace.location_utm_hospital_buffer.plot(ax=ax, color='green', alpha=0.3)
        sim.airspace.location_utm_hospital.plot(ax=ax, color='black')
        #adding vertiports to static plot
        gpd.GeoSeries(sim.vertiports).plot(ax=ax, color='black')
    
    
    static_plot()
    sim.RUN_SIMULATOR(fig, ax, static_plot)

        



    
    







   


