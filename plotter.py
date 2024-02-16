import numpy as np
import matplotlib.pyplot as plt 
from geopandas import GeoSeries
from airspace import Airspace
import uav
import airtrafficcontroller
import simulator


class Plotter:
    def __init__(self, sim_instance):
        self.sim_instance = sim_instance
        self.vertiport_gs = GeoSeries(self.sim_instance.vertiports)
        self.fig, self.ax = plt.subplots()


    def static_plot(self, ):
        self.sim_instance.airspace.location_utm_gdf.plot(ax=self.ax, color='gray', linewidth = 0.6)
        self.sim_instance.airspace.location_utm_hospital_buffer.plot(ax=self.ax, color='green', alpha=0.3)
        self.sim_instance.airspace.location_utm_hospital.plot(ax=self.ax, color='black')
        self.vertiport_gs.plot(ax=self.ax, color = 'black')

    def interactive_plotting_on(self):
        return plt.ion()