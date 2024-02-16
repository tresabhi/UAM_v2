import numpy as np
import matplotlib.pyplot as plt 
from geopandas import GeoSeries
from airspace import Airspace
import uav
import airtrafficcontroller


# Plotter will access the path traces of uavs in airspace, 
# It will have a method that 



plt.ion()
fig, ax = plt.subplots() #have three axes, i)map ii)NMAC count iii)collision count
start_end_array = []


def plot_vertiports(vertiports):
    ax.plot(vertiports)


def plot_static_objects(airspace:Airspace):
    airspace.location_utm_gdf.plot(ax=ax, color='blue', linewidth=0.6)
    airspace.location_utm_hospital_buffer.plot(ax=ax, color='red', alpha=0.3)
    airspace.location_utm_hospital.plot(ax=ax, color='black')

def plotter():
    #TODO - collect path traces of all uavs in airspace
    #TODO - find shortest path_trace length -> loop_len
    #TODO - loop with loop_len and plot the uav path_traces on map 



    #TODO - COMPLETE THE PLOTTING RIGHT NOW, THINK OF EVERYTHING LATER 
    #TODO - plotting 
    #TODO - 