from ..envs.airspace import Airspace

if __name__ == "__main__":
    location_name = "Austin, Texas, USA"
    buffer_radius = 500

    airspace = Airspace(location_name=location_name, buffer_radius=buffer_radius)

    for attribute, value in vars(airspace):
        print(attribute, value)
