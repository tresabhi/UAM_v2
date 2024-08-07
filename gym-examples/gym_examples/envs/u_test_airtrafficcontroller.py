import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.airtrafficcontroller import ATC
from envs.airspace import Airspace


def check_create_n_random_vertiports(num_vertiport):
    atc.create_n_random_vertiports(num_vertiport)
    try:
        assert len(atc.vertiports_in_airspace) == num_vertiport
    except:
        raise AssertionError('length of vp not correct')
    print('VP check passed')
  


if __name__ == "__main__":
    location_name = "Austin, Texas, USA"
    buffer_radius = 500
    airspace = Airspace(location_name=location_name, buffer_radius=buffer_radius)

    atc = ATC(airspace=airspace)
    methods = [
        func
        for func in dir(atc)
        if callable(getattr(atc, func)) and not func.startswith("__")
    ]
    #print(methods)

    # atc.create_n_auto_uavs(5)
    # assert len(atc.auto_uavs_list) == 5
    check_create_n_random_vertiports(10)
    
