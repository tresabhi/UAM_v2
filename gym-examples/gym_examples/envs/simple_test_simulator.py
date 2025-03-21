import numpy as np
from shapely import Point

from space import Space
from uav_v2 import UAV_v2
from controller_static import StaticController
from controller_non_coop import NonCoopController
from controller_non_coop_smooth import NonCoopControllerSmooth
from controller_non_coop_ORCA import ORCA_controller
from dynamics_orca import ORCA_Dynamics
from dynamics_point_mass import PointMassDynamics
from sensor_universal import UniversalSensor


space = Space(10,12, 123)

orca_controller = ORCA_controller(20, np.pi, 5, 0.1)
orca_dynamics = ORCA_Dynamics()
universal_sensor = UniversalSensor(space)

uav1 = UAV_v2(orca_controller, orca_dynamics, universal_sensor, 3, 4, 6)

