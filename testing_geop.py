import numpy as np
import matplotlib.pyplot as plt 
import geopandas as gpd
#from geodatasets import get_path
import geodatasets
import osmnx as ox 
import shapely
#import rasterio
from shapely.geometry import Point
from shapely.geometry import LineString
from matplotlib.animation import FuncAnimation

#* training and testing space - generalise this 
place_name = 'Austin, Texas, USA'

austin_gdf = ox.geocode_to_gdf(place_name) #converts named geocode - 'Austin,Texas' location to gdf 
austin_utm_gdf = ox.projection.project_gdf(austin_gdf) #default projection - UTM projection 
austin_utm_gdf['boundary'] = austin_utm_gdf.boundary #adding column 'boundary' 

austin_hospital = ox.features.features_from_polygon(austin_gdf['geometry'][0], tags={'building':'hospital'})
austin_utm_hospital = ox.projection.project_gdf(austin_hospital)
austin_utm_hospital_buffer = austin_utm_hospital.buffer(500)

#* Spawning inside the boundary of Austin - generalize this 
sample_points_gdf = austin_utm_gdf.sample_points(2) # returns two random points in the map 

#* Define start and end explicitly
sample_points = sample_points_gdf[0]

#* Create a direction vector, that will guide the point
coords = []
for point in sample_points.geoms:
    x = point.x
    y = point.y
    coords.append((x,y))
connecting_line = LineString(coords) #this line is built using 2 sampled points

#* Creating an array of points for the plane to follow 
num_points = 21 

# interpolated_points[Tuple(x,y)]
interpolated_points = []

interpolated_shapely_points = [connecting_line.interpolate(i/num_points, normalized=True) for i in range(1,num_points)]
for i in range(len(interpolated_shapely_points)):
    temp_container = shapely.get_coordinates(interpolated_shapely_points[i])
    x,y = temp_container[0]
    interpolated_points.append((x,y))




#* Existing code for the initial plot setup
fig, ax = plt.subplots(figsize=(20, 10))
austin_utm_gdf.plot(ax=ax, color='blue', linewidth=0.6)
austin_utm_hospital_buffer.plot(ax=ax, color='red', alpha=0.3)
austin_utm_hospital.plot(ax=ax, color='black')
sample_points_gdf.plot(ax=ax, color='black')
gpd.GeoSeries([connecting_line]).plot(ax=ax, color='green')


#* init - used to draw a clear frame 

austin_utm_gdf.plot(ax=ax, color='blue', linewidth=0.6)
austin_utm_hospital_buffer.plot(ax=ax, color='red', alpha=0.3)
austin_utm_hospital.plot(ax=ax, color='black')
sample_points_gdf.plot(ax=ax, color='black')
gpd.GeoSeries([connecting_line]).plot(ax=ax, color='green')
airplane = ax.scatter(interpolated_points[0][0], interpolated_points[0][1])

#* Animation update function
def update(frame):
    airplane.set_offsets((interpolated_points[frame][0], interpolated_points[frame][1]))
    # point = interpolated_points[frame]
    # ax.plot(point.x, point.y, 'x')  # Plotting each point as an 'x'
    # return ax,
    return airplane

#* Create animation
ani = FuncAnimation(fig, update, frames=len(interpolated_points), interval=200, repeat=False)
#                  what is the value of len(interpolated_points)

plt.show()


#! Next steps 

# make an aircraft object which is a point
# use start and end point to calculate bearing 
    # what is bearing, bearing vs heading 
# define a heading/bearing in theta 

# point should move using bearing and step 
# how to identify point is in contact with buffer zone
    # ? use two buffer zones with the point, one as NMAC and another as collision. Intersection with NMAC increments NMAC count
    # ? and intersection with collision increments collision count and takes the drone out from simulation.
# why is theta better than unit vector  
# change the map to include surrounding area of austin 
# find traffic data that is compatible with osmnx and/or geopandas 
