from shapely import Point

from space import Space
from uav_v2 import UAV_v2
from static_controller import StaticController
from point_mass_dynamics import PointMassDynamics
from universal_sensor import UniversalSensor

space = Space()
universal_sensor = UniversalSensor(space=space)
static_controller = StaticController()
pm_dynamics = PointMassDynamics()
uav1 = UAV_v2(controller=static_controller, dynamics=pm_dynamics, sensor=universal_sensor,radius=5)
# uav2 = UAV_v2(controller=)

uav1.assign_start_end(Point(0,0), Point(12,12))

print(f'UAV start: {uav1.start}, end: {uav1.end}')
print(uav1)

for i in range(10):
    print(f'current timestep:{i}, uav current position: {uav1.current_position}')
    action = uav1.get_action(observation=None)
    uav1.step(action=action)
