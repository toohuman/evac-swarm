import random
from mesa import Model
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector

from evac_swarm.agents import RobotAgent, WallAgent, CasualtyAgent
from evac_swarm.building_generator import generate_building_layout

class SwarmExplorerModel(Model):
    """
    The main model representing the building environment and robotic swarm.
    """
    def __init__(
        self,
        width=100,
        height=100,
        robot_count=10,
        casualty_count=3,
        min_room_size=20,
        wall_thickness=1,
        vision_range=15
    ):
        super().__init__()
        
        # Set parameters directly
        self.width = width
        self.height = height
        self.min_room_size = min_room_size
        self.wall_thickness = wall_thickness
        self.vision_range = vision_range
        self.robot_count = robot_count
        self.casualty_count = casualty_count

        # Initialise space
        self.space = ContinuousSpace(self.width, self.height, torus=False)
        print(f"Space type: {type(self.space)}")
        
        # Initialise DataCollector
        self.datacollector = DataCollector(
            model_reporters={"Coverage": lambda m: len(m.coverage_grid)/(self.width*self.height)*100}
        )
        print("DataCollector exists:", hasattr(self, 'datacollector'))
        
        self._next_id = 0  # Add counter for agent IDs
        
        # Generate building layout using a BSP algorithm.
        # Returns a list of wall definitions (each a dict with x, y, width and height)
        self.wall_layout = generate_building_layout(self.width, self.height, self.min_room_size, self.wall_thickness)
        
        # Create Wall agents for each wall piece in the layout.
        for wall_spec in self.wall_layout:
            wall = WallAgent(self._next_id, self, wall_spec)
            self._next_id += 1
            self.register_agent(wall)
            self.space.place_agent(wall, (wall_spec['x'], wall_spec['y']))

        # Define the entry point (deployment operator location).
        # We assume the entry is at the centre of the bottom wall.
        self.entry_point = (self.width / 2, 0 + self.wall_thickness)
        
        # Place Robot agents at the entry point.
        for _ in range(self.robot_count):
            robot = RobotAgent(self._next_id, self, pos=self.entry_point, vision_range=self.vision_range)
            self._next_id += 1
            self.register_agent(robot)
            self.space.place_agent(robot, self.entry_point)
            
        # Randomly place Casualty agents within the building.
        for _ in range(self.casualty_count):
            # For simplicity, sample random positions until one is not colliding with a wall.
            while True:
                pos = (random.uniform(self.wall_thickness, self.width - self.wall_thickness),
                       random.uniform(self.wall_thickness, self.height - self.wall_thickness))
                # Using a simple check: the point should not be inside any wall rectangle.
                if not any(self._point_in_wall(pos, spec) for spec in self.wall_layout):
                    casualty = CasualtyAgent(self._next_id, self, pos)
                    self._next_id += 1
                    self.register_agent(casualty)
                    self.space.place_agent(casualty, pos)
                    break

        # Add coverage tracking
        self.coverage_grid = set()  # Set of (x,y) tuples that have been visited
        
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
        # Update coverage based on robot positions
        for agent in self.agents:
            if isinstance(agent, RobotAgent):
                x, y = agent.pos
                # Add positions within vision range to coverage
                for dx in range(-int(agent.vision_range), int(agent.vision_range) + 1):
                    for dy in range(-int(agent.vision_range), int(agent.vision_range) + 1):
                        if (dx*dx + dy*dy) <= agent.vision_range*agent.vision_range:
                            new_x, new_y = x + dx, y + dy
                            if 0 <= new_x < self.width and 0 <= new_y < self.height:
                                self.coverage_grid.add((int(new_x), int(new_y)))
        
        # Collect data
        self.datacollector.collect(self)
        
        # Step all agents
        for agent in self.agents:
            agent.step() 