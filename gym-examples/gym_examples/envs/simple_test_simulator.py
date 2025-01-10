from shapely import Point

from space import Space
from uav_v2 import UAV_v2
from controller_static import StaticController
from controller_non_coop import NonCoopController
from controller_non_coop_smooth import NonCoopControllerSmooth
from dynamics_point_mass import PointMassDynamics
from sensor_universal import UniversalSensor




space = Space()

universal_sensor = UniversalSensor(space=space)
static_controller = StaticController(0,0)
non_coop_smooth_controller = NonCoopControllerSmooth(10,2)
pm_dynamics = PointMassDynamics()
non_coop_controller = NonCoopController(10,1)

# --- UAV construction --- 
#! need to automate construction of UAVs based on number of vertiports/start-end coordinates
uav1 = UAV_v2(controller=non_coop_smooth_controller, dynamics=pm_dynamics, sensor=universal_sensor,radius=5, nmac_radius=20)
uav2 = UAV_v2(controller=non_coop_controller, dynamics=pm_dynamics, sensor=universal_sensor, radius=5, nmac_radius=20)
# --- UAV start-end assignment ---
uav1.assign_start_end(Point(0,0), Point(120,120))
uav2.assign_start_end(Point(120,120), Point(0,0))
# --- Adding UAVs to space ---
space.set_uav(uav1)
space.set_uav(uav2)


# --- print UAVs to terminal ---
print(f'UAV start: {uav1.start}, end: {uav1.end}')
print(uav1)
print(f'UAV start: {uav2.start}, end: {uav2.end}')
print(uav2)

end_sim = False
for timestep in range(70):
    for uav in space.get_uav_list():
        print(f'current timestep:{timestep}, uav current position: {uav.current_position}')
        uav_state = uav.get_state()
        
        # NMAC 
        is_nmac, nmac_list = uav.sensor.get_nmac(uav)
        if is_nmac:
            print('--- NMAC ---')
            print(f'NMAC detected:{is_nmac}, and NMAC with {nmac_list}\n')
        
        # Collision
        is_collision, collision_uav_ids = uav.sensor.get_collision(uav)
        if is_collision:
            print('---COLLISION---')
            print(f'Collision detected:{is_collision}, and collision with {collision_uav_ids}\n')
            space.remove_uavs_by_id(collision_uav_ids)
            if len(space.uav_list) == 0:
                print('NO more uavs in space')
                end_sim = True
                break

        
        print(f'Observation\n{uav_state}\n')
        # place inside step of gym.env 
        observation = uav.get_obs()
        uav_mission_complete_status = uav.get_mission_status()
        uav.set_mission_complete_status(uav_mission_complete_status)
        action = uav.get_action(observation=observation)
        uav.dynamics.update(uav, action)
    if end_sim:
        print('--- END SIMULATION ---')
        break





#TODO - need to create a random assignment type of start end vertiport coordinates
space = Space(max_uavs=10, max_vertiports=10)

universal_sensor = UniversalSensor(space=space)
static_controller = StaticController(0,0)
non_coop_smooth_controller = NonCoopControllerSmooth(10,2)
non_coop_controller = NonCoopController(10,1)
pm_dynamics = PointMassDynamics()
universal_sensor = UniversalSensor(space=space)

space.create_vertiports()
#TODO - create_uav(has_agent:bool) will use num_vertiports to create one less UAV than the number of vertiports, the last UAV will be RL agent UAV 
space.create_uav(6, UAV_v2,non_coop_controller,pm_dynamics,universal_sensor,5,20)
space.assign_vertiports()

print(space.get_vertiport_list())

for uav in space.get_uav_list():
    print(uav.start, uav.end)







