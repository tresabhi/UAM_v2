#mapped_env_util.py
from gymnasium.spaces import Dict, Box, Discrete
import numpy as np
import math 
import geopandas as gpd
from matplotlib import pyplot as plt 
from matplotlib.patches import Circle, FancyArrowPatch
from auto_uav_v2 import Auto_UAV_v2



#### REMOVE BLOCK START ####
def choose_obs_space_constructor(obs_space_string:str):
    '''This helper method uses the argument to return correct obs_space constructor for gym env
        Args:
            obs_space_string
        
        Returns:
            function (object)'''
    if obs_space_string == 'LSTM-A2C':
        return obs_space_seq
    elif obs_space_string == 'GNN-A2C':
        return obs_space_graph
    elif obs_space_string == 'UAM_UAV':
        return obs_space_uam


# Define sequential observation space for LSTM
def obs_space_seq(max_number_other_agents_observed):
    '''Gym observation space for LSTM-A2C model'''
    return Dict(
    {
        "no_other_agents": Box(
            low=0, high=max_number_other_agents_observed, shape=()
        ),
        "dist_goal": Box(low=0, high=10000, shape=(), dtype=np.float32),
        "heading_ego_frame": Box(low=-180, high=180, shape=(), dtype=np.float32),
        "current_speed": Box(low=0, high=50, shape=(), dtype=np.float32),
        "radius": Box(low=0, high=20, shape=(), dtype=np.float32),  # UAV size
        # Static object detection
        "static_collision_detected": Box(low=0, high=1, shape=(), dtype=np.int32),
        "distance_to_restricted": Box(low=0, high=10000, shape=(), dtype=np.float32),
        #FIX:  add another key:value for static object, the heading of static_obj
        
        # Other agent data
        "other_agent_state": Box(  # p_parall, p_orth, v_parall, v_orth, other_agent_radius, combined_radius, dist_2_other
            low=np.full(
                (max_number_other_agents_observed, 7), -np.inf
            ),
            high=np.full(
                (max_number_other_agents_observed, 7), np.inf
            ),
            shape=(
                max_number_other_agents_observed,
                7,
            ),
            dtype=np.float32,
        ),
    }
)

# Define graph observation space for GNN
def obs_space_graph(max_number_other_agents_observed):
    '''Gym obs space for GNN(and variants)-A2C model'''
    return Dict(
        {
            "num_other_agents": Box(low=0, high=100, shape=(), dtype=np.int64),
            "agent_dist_to_goal": Box(
                low=0, high=np.inf, shape=(), dtype=np.float32
            ),
            "agent_end_point": Box(
                low=np.array([-np.inf, -np.inf]),
                high=np.array([np.inf, np.inf]),
                shape=(2,),
                dtype=np.float32,
            ),
            "agent_current_position": Box(
                low=np.array([-np.inf, -np.inf]),
                high=np.array([np.inf, np.inf]),
                shape=(2,),
                dtype=np.float32,
            ),
            "static_collision_detected": Box(low=0, high=1, shape=(), dtype=np.int32),
            "distance_to_restricted": Box(low=0, high=10000, shape=(), dtype=np.float32),
            "graph_feat_matrix": Box(
                low=np.full(
                    (max_number_other_agents_observed + 1, 5), -np.inf
                ),
                high=np.full((max_number_other_agents_observed + 1, 5), np.inf),
                shape=(max_number_other_agents_observed + 1, 5),
                dtype=np.float32,
            ),
            "edge_index": Box(
                low=0,
                high=max_number_other_agents_observed,
                shape=(2, max_number_other_agents_observed),
                dtype=np.int64,
            ),
            "edge_attr": Box(
                low=0,
                high=np.inf,
                shape=(
                    max_number_other_agents_observed,
                    1,
                ),
                dtype=np.float32,
            ),
            "mask": Box(
                low=0,
                high=1,
                shape=(max_number_other_agents_observed + 1,),
                dtype=np.float32,
            ),
        }
    )



def obs_space_uam(auto_uav):
    '''Obs space for one intruder and restricted area'''
    return Dict(
            {
                # agent ID as integer
                "agent_id": Box(
                    low=0,
                    high=np.iinfo(np.int64).max,
                    shape=(1,),
                    dtype=np.int64,  #! find if it is possible to create ids that take less space
                ),
                # agent speed
                "agent_speed": Box(  #!need to rename velocity -> speed
                    low=-auto_uav.max_speed,  # agent's speed #! need to check why this is negative
                    high=auto_uav.max_speed,
                    shape=(1,),
                    dtype=np.float64,
                ),
                # agent deviation
                "agent_deviation": Box(
                    low=-360,
                    high=360,
                    shape=(1,),
                    dtype=np.float64,  # agent's heading deviation #!should this be -180 to 180, if yes then this needs to be corrected to -180 to 180
                ),
                # intruder detection
                "intruder_detected": Discrete(
                    2  # 0 for no intruder, 1 for intruder detected
                ),
                # intruder id
                "intruder_id": Box(
                    low=0,
                    high=np.iinfo(np.int64).max,
                    shape=(1,),
                    dtype=np.int64,  #! find if it is possible to create ids that take less space
                ),
                # distance to intruder
                "distance_to_intruder": Box(
                    low=0,
                    high=auto_uav.detection_radius,
                    shape=(1,),
                    dtype=np.float64,
                ),
                # Relative heading of intruder #!should this be corrected to -180 to 180,
                "relative_heading_intruder": Box(
                    low=-360, high=360, shape=(1,), dtype=np.float64
                ),
                "intruder_current_heading": Box(
                    low=-180, high=180, shape=(1,), dtype=np.float64
                ),  # Intruder's heading
                
                # restricted airspace
                "restricted_airspace_detected":Discrete(
                    2 
                ),
                # distance to airspace 
                "distance_to_restricted_airspace": Box(
                    low=0,
                    high=1000,
                    shape=(1,),
                    dtype=np.float64,
                ),
            }
        )


def get_obs_uam_uav(self):  
    '''
    Internal method to collect observation for the agent/auto_uav
    Return: dict(observation dictionary)
    '''
    """
    Gets the observation space of the agent

    Args:
        agent_id (str): The ID of the target UAV

    Returns:
        obs (dict): The observation space of the agent
            agent_id
            agent_speed
            agent_deviation
            intruder_detected
            intruder_id
            distance_to_intruder
            relative_heading_intruder
            intruder_heading
    """

    #! this observation does not include the static objects
    #! add observation for static objects  
    agent_id = np.array([self.agent.id]) #
    agent_speed = self.agent.current_speed
    agent_deviation = self.agent.current_heading - self.agent.final_heading
    
    #FIX: self.agent.sensor.get_uav_detection(self.agent) -> could return None, when no intruder
    intruders_info = self.agent.sensor.get_uav_detection(self.agent)
    #FIX: define a way to choose intruder_index - closest or some other metric
    intruder_info = intruders_info[intruder_index] 

    if intruder_info:
        intruder_detected = 1
        intruder_id = np.array([intruder_info['other_uav_id']])
        intruder_pos = intruder_info['other_uav_current_position']
        intruder_heading = np.array([intruder_info['other_uav_current_heading']])
        distance_to_intruder = np.array(
            [self.agent.current_position.distance(intruder_info['other_uav_current_position'])]
        )
        relative_heading_intruder = np.array(
            [self.agent.current_heading - intruder_info['other_uav_current_heading']]
        )
    else:

        intruder_detected = 0
        intruder_id = np.array([0])
        distance_to_intruder = np.array([0])
        relative_heading_intruder = np.array([0])
        intruder_heading = np.array([0])

    #! restricted airspace detected - should this be detection/nmac
    # if the detection area of uav intersects with a building's detection area,
    # we should do the following -
    # 1) if intersection is detected for 'detection' argument ->
    #                   there should be a small penalty
    #                   based on distance between the uav_footprint and the actual building
    # 2) if there is no building detected the penalty should be zero
    # 3) just like intruder_detected in obs
    #    there will be restricted_airspace detected in obs
    # 4) restricted_airspace will have 0 for not detected 1 for detected,
    #    and if detected - distance will be added to the obs, if not detected distance is 0,
    #    this will be handled by the reward function collect obs and based on restricted_airspace (y/n)
    #    penatly is something or 0.

    #! use shapely method shapely.ops.nearest_points(geom1, geom2)->list; return a list of two points point_geom1, point_geom2
    
    if self.agent.sensor.get_ra_detection()[0] == True:
        ra_detected = 1
        ra_distance = self.agent.sensor.get_ra_detection()[1]['distance']
        ra_heading = self.agent.sensor.get_ra_detection()[1]['ra_heading'] 
    else:
        ra_detected = 0
        ra_distance = 0
        ra_heading = 0



    

    observation = {
        "agent_id": agent_id,
        "agent_speed": agent_speed,
        "agent_deviation": agent_deviation,
        "intruder_detected": intruder_detected,
        "intruder_id": intruder_id,
        "distance_to_intruder": distance_to_intruder,
        "relative_heading_intruder": relative_heading_intruder,
        "intruder_current_heading": intruder_heading,
        "ra_detected": ra_detected,
        "ra_distance": ra_distance,
        "ra_heading": ra_heading
    }

    return observation


#TODO: remove the code above and rename script to utils_render_mapped_env.py
#### REMOVE BLOCK END ####


#### RENDERING UTILS ####

def create_animation(self, env_time_step):
    """Create an animation of the environment."""
    # Ensure data exists
    if len(self.df) == 0:
        print("No animation data available")
        return None
    
    try:
        from matplotlib.animation import FuncAnimation
        
        # Pre-process data for animation to avoid df lookups during animation
        self.animation_data = {}
        self.animation_time_steps = sorted(self.df['current_time_step'].unique())
        
        # Group data by time step
        for time_step in self.animation_time_steps:
            step_data = self.df[self.df['current_time_step'] == time_step]
            self.animation_data[time_step] = []
            
            # Extract and store UAV data for this time step
            for _, row in step_data.iterrows():
                uav_data = {
                    'id': row['uav_id'],
                    'position': row['current_position'],
                    'heading': row['current_heading'],
                    'final_heading': row['final_heading'],
                    'is_auto': isinstance(row['uav'], Auto_UAV_v2),
                    'uav': row['uav']
                }
                self.animation_data[time_step].append(uav_data)
        
        # Create trajectories by UAV ID
        self.animation_trajectories = {}
        for time_step in self.animation_time_steps:
            for uav_data in self.animation_data[time_step]:
                uav_id = uav_data['id']
                position = uav_data['position']
                
                if uav_id not in self.animation_trajectories:
                    self.animation_trajectories[uav_id] = []
                
                # Store time step and position
                self.animation_trajectories[uav_id].append((time_step, position))
                
        # Create a new figure specifically for animation
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Function to update animation frame
        def animate(frame_index):
            if frame_index >= len(self.animation_time_steps):
                return []
                
            time_step = self.animation_time_steps[frame_index]
            ax.clear()
            
            # Draw static assets
            self.render_static_assets(ax)
            
            # Draw vertiports with larger markers
            for vertiport in self.space.get_vertiport_list():
                ax.plot(vertiport.x, vertiport.y, 'gs', markersize=12)
            
            # Draw UAVs at this time step
            current_uav_ids = []
            for uav_data in self.animation_data[time_step]:
                pos = uav_data['position']
                heading = uav_data['heading']
                final_heading = uav_data['final_heading']
                is_auto = uav_data['is_auto']
                uav_id = uav_data['id']
                uav_obj = uav_data['uav']
                
                current_uav_ids.append(uav_id)
                
                # Set colors based on UAV type
                detection_color = '#0278c2' if is_auto else 'green'
                nmac_color = '#FF7F50' if is_auto else 'orange'
                body_color = '#0000A0' if is_auto else 'blue'
                
                # Draw detection radius
                detection_circle = Circle((pos.x, pos.y),
                                    uav_obj.detection_radius,
                                    fill=False, color=detection_color, alpha=0.3, linewidth=2)
                ax.add_patch(detection_circle)
                
                # Draw NMAC radius
                nmac_circle = Circle((pos.x, pos.y),
                                uav_obj.nmac_radius,
                                fill=False, color=nmac_color, alpha=0.4, linewidth=2)
                ax.add_patch(nmac_circle)
                
                # Draw UAV body
                body_circle = Circle((pos.x, pos.y),
                                uav_obj.radius,
                                fill=True, color=body_color, alpha=0.7)
                ax.add_patch(body_circle)
                
                # Draw current heading arrow
                heading_length = uav_obj.radius * 5  # Make longer for visibility
                dx = heading_length * np.cos(heading)
                dy = heading_length * np.sin(heading)
                arrow = FancyArrowPatch((pos.x, pos.y),
                                (pos.x + dx, pos.y + dy),
                                color='black',
                                arrowstyle='->',
                                mutation_scale=10,
                                linewidth=2.5)
                ax.add_patch(arrow)
                
                # Draw final heading (reference direction) with thicker line
                ref_length = uav_obj.radius * 4
                dx_ref = ref_length * np.cos(final_heading)
                dy_ref = ref_length * np.sin(final_heading)
                ref_color = 'purple' if is_auto else 'red'
                ref_arrow = FancyArrowPatch((pos.x, pos.y),
                                    (pos.x + dx_ref, pos.y + dy_ref),
                                    color=ref_color,
                                    arrowstyle='->',
                                    mutation_scale=7.5,
                                    linewidth=2,
                                    alpha=0.6)
                ax.add_patch(ref_arrow)
                
                # Draw start-end connection with thicker line
                if hasattr(uav_obj, 'start') and hasattr(uav_obj, 'end'):
                    line_color = 'blue' if is_auto else 'green'
                    ax.plot([uav_obj.start.x, uav_obj.end.x],
                        [uav_obj.start.y, uav_obj.end.y],
                        '--', color=line_color, alpha=0.6, linewidth=2.0)
            
            # Draw trajectories up to this time step
            # This ensures we don't show trajectories for removed UAVs
            for uav_id in current_uav_ids:
                if uav_id in self.animation_trajectories:
                    # Get trajectory points up to current time step
                    traj_points = [(t, p) for t, p in self.animation_trajectories[uav_id] if t <= time_step]
                    
                    if len(traj_points) > 1:
                        # Extract positions
                        positions = [p for _, p in traj_points]
                        xs = [p.x for p in positions]
                        ys = [p.y for p in positions]
                        
                        # Determine color based on UAV type
                        is_auto = any(d['is_auto'] for d in self.animation_data[time_step] if d['id'] == uav_id)
                        line_color = '#0000A0' if is_auto else 'blue'
                        
                        # Draw trajectory line with thicker width
                        ax.plot(xs, ys, '-', linewidth=2.5, alpha=0.6, color=line_color)
            
            # Calculate proper plot limits to see the whole map
            vp_x_coords = [v.x for v in self.space.get_vertiport_list()]
            vp_y_coords = [v.y for v in self.space.get_vertiport_list()]
            
            # Add UAV positions
            uav_x_coords = [uav_data['position'].x for uav_data in self.animation_data[time_step]]
            uav_y_coords = [uav_data['position'].y for uav_data in self.animation_data[time_step]]
            
            # Combine for full area
            all_x_coords = vp_x_coords + uav_x_coords
            all_y_coords = vp_y_coords + uav_y_coords
            
            # Add restricted areas dimensions
            for tag_value in self.airspace.location_tags.keys():
                # Get bounds of restricted areas
                restricted_bounds = self.airspace.location_utm[tag_value].bounds
                if len(restricted_bounds) > 0:
                    for bound in restricted_bounds.values:
                        if len(bound) >= 4:  # minx, miny, maxx, maxy
                            all_x_coords.extend([bound[0], bound[2]])
                            all_y_coords.extend([bound[1], bound[3]])
            
            # Set limits with margin
            if all_x_coords and all_y_coords:
                x_min, x_max = min(all_x_coords), max(all_x_coords)
                y_min, y_max = min(all_y_coords), max(all_y_coords)
                
                # Add margin to ensure all elements are visible
                margin = max(500, (x_max - x_min) * 0.1)
                ax.set_xlim(x_min - margin, x_max + margin)
                ax.set_ylim(y_min - margin, y_max + margin)
            
            ax.set_title(f'UAM Simulation - Step {time_step}')
            ax.set_aspect('equal')
            
            return []
        
        # Create animation object
        frames = min(env_time_step, len(self.animation_time_steps))
        if frames == 0:
            print("No frames to animate")
            return None
            
        ani = FuncAnimation(
            fig, 
            animate, 
            frames=frames,
            interval=200,
            blit=False
        )
        
        return ani
        
    except Exception as e:
        print(f"Error creating animation: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_animation(self, animation_obj, file_name):
    """Save animation to a file with optimized quality and compatibility."""
    if animation_obj is None:
        print("No animation to save")
        return
        
    try:
        # Save as MP4 first with optimized settings
        try:
            print(f"Saving animation to {file_name}.mp4...")
            from matplotlib.animation import FFMpegWriter
            
            # Try to use a writer with optimized settings
            writer = FFMpegWriter(
                fps=10,  # Higher fps for smoother playback
                metadata=dict(title='UAM Simulation'),
                bitrate=5000,  # Higher bitrate for better quality
                extra_args=['-vcodec', 'mpeg4', 
                        '-pix_fmt', 'yuv420p',
                        '-q:v', '3']  # Quality value - lower is better quality (1-31)
            )
            
            # First attempt with quality settings
            try:
                animation_obj.save(
                    f"{file_name}.mp4",
                    writer=writer,
                    dpi=200  # Higher DPI for better quality
                )
                print("MP4 saved successfully with high quality settings!")
            except Exception as e:
                print(f"High quality MP4 save failed: {e}")
                
                # Fallback to simpler settings
                try:
                    print("Trying with simpler MP4 settings...")
                    animation_obj.save(
                        f"{file_name}.mp4",
                        writer='ffmpeg',
                        fps=8,
                        dpi=150
                    )
                    print("MP4 saved successfully with basic settings!")
                except Exception as e:
                    print(f"Basic MP4 save failed: {e}")
                    
                    # Try with minimal settings
                    try:
                        print("Trying with minimal MP4 settings...")
                        animation_obj.save(
                            f"{file_name}.mp4",
                            writer='ffmpeg',
                            fps=5,
                            dpi=100
                        )
                        print("MP4 saved successfully with minimal settings!")
                    except Exception as e:
                        print(f"Minimal MP4 save failed: {e}")
        except Exception as mp4_error:
            print(f"MP4 saving failed completely: {mp4_error}")
            
        # Save as GIF (as backup)
        try:
            print(f"Saving animation to {file_name}.gif...")
            from matplotlib.animation import PillowWriter
            
            # Use higher quality settings for GIF
            animation_obj.save(
                f"{file_name}.gif",
                writer=PillowWriter(fps=8),
                dpi=150  # Higher DPI for better quality
            )
            print("GIF saved successfully!")
        except Exception as gif_error:
            print(f"GIF save failed: {gif_error}")
            
            # Try with minimal settings
            try:
                print("Trying with minimal GIF settings...")
                animation_obj.save(
                    f"{file_name}.gif",
                    writer=PillowWriter(fps=5),
                    dpi=100
                )
                print("GIF saved successfully with minimal settings!")
            except Exception as e:
                print(f"Minimal GIF save failed: {e}")
                
    except Exception as e:
        print(f"Error in animation saving: {e}")
        import traceback
        traceback.print_exc()


def add_data(self, uav):
    """Add UAV data to animation dataframe."""
    self.df = self.df._append(
        {
            "current_time_step": self.current_time_step,
            "uav_id": uav.id,
            "uav": uav,
            "current_position": uav.current_position,
            "current_heading": uav.current_heading,
            "final_heading": math.atan2(uav.end.y - uav.current_position.y, 
                                    uav.end.x - uav.current_position.x),
        },
        ignore_index=True,
    )

def render_static_assets(self, ax):
    """Render the static assets of the environment (map, restricted areas)."""
    # Draw map boundaries
    self.airspace.location_utm_gdf.plot(ax=ax, color="gray", linewidth=0.6)
    
    # Draw restricted areas
    for tag_value in self.airspace.location_tags.keys():
        # Draw actual restricted areas
        self.airspace.location_utm[tag_value].plot(ax=ax, color="red", alpha=0.7)
        # Draw buffer zones
        self.airspace.location_utm_buffer[tag_value].plot(ax=ax, color="orange", alpha=0.3)
    
    # Draw vertiports
    vertiport_points = [v for v in self.space.get_vertiport_list()]
    if vertiport_points:
        gpd.GeoSeries(vertiport_points).plot(ax=ax, color="black", markersize=10)
