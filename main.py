import airspace
import uav
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import geopandas as gpd
import time


if __name__ == '__main__':
    location_name = 'Austin, Texas, USA' #pass to airspace
    airspace = airspace.Airspace(location_name=location_name)
    start_point, end_point = airspace.get_random_start_end_points()
    
    uav1 = uav.UAV(start_point, end_point,)

    path_trace = []

    while ((np.abs(uav1.current_position.x - uav1.end_point.x)>5) and (np.abs(uav1.current_position.y - uav1.end_point.y)>5)):
        uav1.step()
        path_trace.append(uav1.current_position)
    
    
    # print('Start Point:', start_point, 'End Point:', end_point)
    # print("Flight completed.")
    # print(len(path_trace))
    # print(type(path_trace[0]))
    plt.ion() 
    fig, ax = plt.subplots()
    start_end_point = gpd.GeoSeries([start_point, end_point])
    # airspace.location_utm_gdf.plot(ax=ax, color='blue', linewidth=0.6)
    # airspace.location_utm_hospital_buffer.plot(ax=ax, color='red', alpha=0.3)
    # airspace.location_utm_hospital.plot(ax=ax, color='black')
    def static_plot():
        airspace.location_utm_gdf.plot(ax=ax, color='blue', linewidth=0.6)
        airspace.location_utm_hospital_buffer.plot(ax=ax, color='red', alpha=0.3)
        airspace.location_utm_hospital.plot(ax=ax, color='black')
        start_end_point.plot(ax=ax, color='black')
    
    #Plot start and end point
    # start_end_point = gpd.GeoSeries([start_point, end_point])
    # start_end_point.plot(ax=ax, color='black')
    
     
    #* Using FuncAnimation to animate the path of the UAV
    # def animate(frame):
    #     gpd.GeoSeries([path_trace[frame]]).plot(ax=ax, color='red', alpha=0.3)
    
    # ani = FuncAnimation(fig, animate, frames=len(path_trace), interval=100, repeat=False)
    # plt.show()

    #* Using for loop to animate the path of the UAV - without trace
    static_plot() 
    for i in range(len(path_trace)):
        plt.cla()
        static_plot()
        gpd.GeoSeries([path_trace[i]]).plot(ax=ax, color='red', alpha=0.3)
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.1)


    
    







   


