

from shapely import Point
from typing import List


class Vertiport:
    def __init__(self, location: Point, uav_list: list = []) -> None:
        self.id = id(self)
        self.location = location
        self.uav_list: List = uav_list
        self.landing_takeoff_capacity = 4

    def __repr__(
        self,
    ) -> str:
        return "Vertiport({location}, {uav_list})".format(
            location=self.location, uav_list=self.uav_list
        )
    

    def landing_queue(self):
        pass


    def takeoff_queue(self):
        pass


    def get_uav_list():
        pass

    