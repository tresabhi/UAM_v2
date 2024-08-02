import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.airspace import Airspace

if __name__ == "__main__":
    location_name = "Austin, Texas, USA"
    buffer_radius = 500

    airspace = Airspace(location_name=location_name, buffer_radius=buffer_radius)
    airspace_attributes = vars(airspace)

    for attribute, value in airspace_attributes.items():
        print(f"Attribute: '{attribute}', Type: '{type(attribute)}'")
