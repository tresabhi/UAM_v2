import math
from typing import List, Tuple

def get_coords(num_co_ords: int, distance: float) -> List[Tuple[float, float]]:
    """
    Return coordinates evenly distributed in a circle, where consecutive coordinate pairs
    are equal distance away from one another.
    
    Args:
        num_co_ords (int): The number of coordinates (must be even).
        distance (float): The distance between consecutive coordinates.
        
    Returns:
        List[Tuple[float, float]]: List of (x, y) tuples.
    """
    if num_co_ords % 2 != 0:
        raise RuntimeError('num_co_ords argument requires an even number.')
    
    # Angle between consecutive points in the polygon
    angle_between_points = (2 * math.pi) / num_co_ords
    
    # Radius of the circle to ensure the given distance
    radius = distance / (2 * math.sin(math.pi / num_co_ords))
    
    co_ords_list = []
    for i in range(num_co_ords):
        angle = i * angle_between_points
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        co_ords_list.append((round(x,2), round(y,2)))
    
    return co_ords_list