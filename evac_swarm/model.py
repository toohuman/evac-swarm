import random
import numpy as np
from mesa import Model
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector
from mesa.experimental.cell_space import PropertyLayer
from mesa.experimental.devs import ABMSimulator

from evac_swarm.agents import RobotAgent, WallAgent, CasualtyAgent
from evac_swarm.building_generator import generate_building_layout
from evac_swarm.space import HybridSpace

class SwarmExplorerModel(Model):
    """
    The main model representing the building environment and robotic swarm.
    """
    def __init__(
        self,
        width=15,
        height=15,
        robot_count=10,
        casualty_count=3,
        min_room_size=3,
        wall_thickness=0.3,
        vision_range=3,
        grid_size=100,
        seed=None,
        simulator: ABMSimulator = None,
    ):
        # Initialize random number generator with seed
        super().__init__()
        self.random = random.Random()
        
        # Set parameters directly
        self.width = width
        self.height = height
        self.min_room_size = min_room_size
        self.wall_thickness = wall_thickness
        self.vision_range = vision_range
        self.robot_count = robot_count
        self.casualty_count = casualty_count
        self.grid_size = grid_size

        # Initialize hybrid space
        self.space = HybridSpace(width, height, grid_size=grid_size, torus=False)
        
        # Generate building layout with new seed
        wall_layout = generate_building_layout(
            width, height, 
            min_room_size, 
            wall_thickness,
            rng=self.random  # Use new seed for building
        )
        
        # Add walls to both representations
        for wall in wall_layout:
            self.space.add_wall(wall)
        
        # Add coverage tracking using grid coordinates
        self.coverage_grid = np.zeros((grid_size, grid_size), dtype=bool)
        
        # Calculate total accessible cells (non-wall cells)
        self.total_accessible_cells = grid_size * grid_size - np.sum(self.space.wall_grid)

        # Update DataCollector
        self.datacollector = DataCollector(
            model_reporters={
                "Coverage": lambda m: (np.sum(m.coverage_grid) / m.total_accessible_cells) * 100
            }
        )
        
        self._next_id = 0  # Add counter for agent IDs
        
        # Define the entry point (deployment operator location).
        # We assume the entry is at the centre of the bottom wall.
        self.entry_point = (self.width / 2, 0 + self.wall_thickness)
        
        # Place Robot agents at the entry point.
        for _ in range(self.robot_count):
            robot = RobotAgent(self._next_id, self, pos=self.entry_point, vision_range=self.vision_range)
            self._next_id += 1
            self.register_agent(robot)
            self.space.place_agent(robot, self.entry_point)  # Continuous coordinates
            
        # Randomly place Casualty agents within the building.
        for _ in range(self.casualty_count):
            while True:
                pos = (
                    self.random.uniform(self.wall_thickness, self.width - self.wall_thickness),
                    self.random.uniform(self.wall_thickness, self.height - self.wall_thickness)
                )
                if not any(self._point_in_wall(pos, spec) for spec in wall_layout):
                    casualty = CasualtyAgent(self._next_id, self, pos)
                    self._next_id += 1
                    self.register_agent(casualty)
                    self.space.place_agent(casualty, pos)
                    break

        self.running = True

        self.simulator = simulator
        if self.simulator is not None:
            self.simulator.setup(self)  # Ensure the simulator is set up on the model instance.

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
                # Convert vision range to grid coordinates
                vision_grid_range = int(agent.vision_range * self.grid_size / self.width)
                
                # Get grid position
                grid_x, grid_y = self.space.continuous_to_grid(x, y)
                
                # Add positions within vision range to coverage
                for dx in range(-vision_grid_range, vision_grid_range + 1):
                    for dy in range(-vision_grid_range, vision_grid_range + 1):
                        if (dx*dx + dy*dy) <= vision_grid_range*vision_grid_range:
                            new_x, new_y = grid_x + dx, grid_y + dy
                            if (0 <= new_x < self.grid_size and 
                                0 <= new_y < self.grid_size and 
                                not self.space.wall_grid[new_y, new_x]):
                                self.coverage_grid[new_y, new_x] = True
        
        # Collect data
        self.datacollector.collect(self)
        
        # Step all agents
        for agent in self.agents:
            agent.step() 