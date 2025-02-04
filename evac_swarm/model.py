import random
from evac_swarm import Model
from evac_swarm.space import ContinuousSpace
from evac_swarm.time import RandomActivation

from agents import RobotAgent, WallAgent, CasualtyAgent
from building_generator import generate_building_layout

class BuildingModel(Model):
    """
    The main model representing the building environment and robotic swarm.
    """
    def __init__(self, width=100, height=100, robot_count=10, casualty_count=3,
                 min_room_size=20, wall_thickness=1, vision_range=15):
        self.width = width
        self.height = height
        # Continuous space: x in [0,width], y in [0,height]
        self.space = ContinuousSpace(width, height, torus=False)
        self.schedule = RandomActivation(self)
        self.vision_range = vision_range
        
        # Generate building layout using a BSP algorithm.
        # Returns a list of wall definitions (each a dict with x, y, width and height)
        self.wall_layout = generate_building_layout(width, height, min_room_size, wall_thickness)
        
        # Create Wall agents for each wall piece in the layout.
        for wall_spec in self.wall_layout:
            wall = WallAgent(self.next_id(), self, wall_spec)
            self.schedule.add(wall)
            self.space.place_agent(wall, (wall_spec['x'], wall_spec['y']))

        # Define the entry point (deployment operator location).
        # We assume the entry is at the centre of the bottom wall.
        self.entry_point = (width / 2, 0 + wall_thickness)
        
        # Place Robot agents at the entry point.
        for _ in range(robot_count):
            robot = RobotAgent(self.next_id(), self, pos=self.entry_point, vision_range=vision_range)
            self.schedule.add(robot)
            self.space.place_agent(robot, self.entry_point)
            
        # Randomly place Casualty agents within the building.
        for _ in range(casualty_count):
            # For simplicity, sample random positions until one is not colliding with a wall.
            while True:
                pos = (random.uniform(wall_thickness, width - wall_thickness),
                       random.uniform(wall_thickness, height - wall_thickness))
                # Using a simple check: the point should not be inside any wall rectangle.
                if not any(self._point_in_wall(pos, spec) for spec in self.wall_layout):
                    casualty = CasualtyAgent(self.next_id(), self, pos)
                    self.schedule.add(casualty)
                    self.space.place_agent(casualty, pos)
                    break

        self.running = True

    def _point_in_wall(self, point, wall_spec):
        """Check whether a point is inside a wall rectangle defined by wall_spec."""
        x, y = point
        wx, wy = wall_spec['x'], wall_spec['y']
        half_w = wall_spec['width'] / 2
        half_h = wall_spec['height'] / 2
        return (wx - half_w <= x <= wx + half_w) and (wy - half_h <= y <= wy + half_h)

    def step(self):
        """Advance the model by one step."""
        self.schedule.step() 