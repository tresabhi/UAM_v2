from shapely import Point

from space import Space
from uav_v2 import UAV_v2
from controller_static import StaticController
from controller_non_coop import NonCoopController
from dynamics_point_mass import PointMassDynamics
from sensor_universal import UniversalSensor

space = Space()

universal_sensor = UniversalSensor(space=space)
static_controller = StaticController(0,0)
pm_dynamics = PointMassDynamics()
non_coop_controller = NonCoopController(10,1)
uav1 = UAV_v2(controller=static_controller, dynamics=pm_dynamics, sensor=universal_sensor,radius=5)
uav2 = UAV_v2(controller=non_coop_controller, dynamics=pm_dynamics, sensor=universal_sensor, radius=5)

uav1.assign_start_end(Point(0,0), Point(12,12))
uav2.assign_start_end(Point(5,5), Point(15,15))
space.set_uav(uav1)
space.set_uav(uav2)
print(f'UAV start: {uav1.start}, end: {uav1.end}')
print(uav1)
print(f'UAV start: {uav2.start}, end: {uav2.end}')
print(uav2)

for i in range(100):
    for uav in space.get_uav_list():
        print(f'current timestep:{i}, uav current position: {uav.current_position}')
        uav_state = uav.get_state()
        print(f'Observation\n{uav_state}\n')
        observation = uav.get_obs()
        uav_mission_complete_status = uav.get_mission_status()
        uav.set_mission_complete_status(uav_mission_complete_status)
        action = uav.get_action(observation=observation)
        uav.dynamics.update(uav, action)