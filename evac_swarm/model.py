import random
import numpy as np
from mesa import Model
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector
from mesa.experimental.cell_space import PropertyLayer
from mesa.experimental.devs import ABMSimulator
from scipy.ndimage import binary_dilation

from evac_swarm.agents import RobotAgent, WallAgent, CasualtyAgent
from evac_swarm.building_generator import generate_building_layout
from evac_swarm.space import HybridSpace

def bresenham(x0, y0, x1, y1):
    """
    Vectorised Bresenham's Algorithm using NumPy.
    Returns a numpy array of integer coordinates along the line from (x0, y0) to (x1, y1).
    """
    dx = x1 - x0
    dy = y1 - y0
    n = int(max(abs(dx), abs(dy)))
    if n == 0:
        return np.array([[x0, y0]])
    t = np.linspace(0, 1, n + 1)
    xs = np.rint(x0 + dx * t).astype(int)
    ys = np.rint(y0 + dy * t).astype(int)
    return np.column_stack((xs, ys))

class SwarmExplorerModel(Model):
    """
    The main model representing the building environment and robotic swarm.
    """
    def __init__(
        self,
        width=20,
        height=20,
        robot_count=10,
        casualty_count=5,
        min_room_size=11,
        wall_thickness=0.3,
        vision_range=3,
        grid_size=100,
        seed=None,
        use_seed=False,
        simulator: ABMSimulator = None,
    ):
        if type(seed) == dict:
            seed = seed['value']
        if not use_seed:
            seed = None
        super().__init__(seed=seed)

        # # Initialize random number generator with seed
        # self.random = random.Random(seed) if seed is not None else random.Random()
        
        # Set parameters directly
        self.width = float(width)
        self.height = float(height)
        self.min_room_size = float(min_room_size)
        self.wall_thickness = float(wall_thickness)
        self.robot_count = int(robot_count)
        self.casualty_count = int(casualty_count)
        self.vision_range = int(vision_range)
        self.grid_size = grid_size

        self.simulator = simulator
        if self.simulator is not None:
            self.simulator.setup(self)  # Ensure the simulator is set up on the model instance.
        
        # Initialize hybrid space
        self.space = HybridSpace(self.width, self.height, grid_size=self.grid_size, torus=False)
        
        # Generate building layout with new seed
        wall_layout = generate_building_layout(
            self.width, self.height, 
            self.min_room_size, 
            self.wall_thickness,
            rng=self.random  # Use new seed for building
        )
        
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
        
        # Add walls to both representations
        for wall in wall_layout:
            self.space.add_wall(wall)
            # Create a WallAgent for each wall
            wall_agent = WallAgent(self._next_id, self, wall_spec=wall)
            self._next_id += 1
            self.register_agent(wall_agent)
        
        # Define the entry point (deployment operator location).
        # We assume the entry is at the centre of the bottom wall.
        self.entry_point = (self.width / 2, 1 + self.wall_thickness)
        
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

        # Precompute the coverage offsets for the shared vision range
        vision_grid_range = int(vision_range * grid_size / width)
        r = np.arange(-vision_grid_range, vision_grid_range + 1)
        dx, dy = np.meshgrid(r, r, indexing='xy')
        circle_mask = (dx**2 + dy**2) <= vision_grid_range**2
        self.coverage_offsets = np.stack((dx[circle_mask], dy[circle_mask]), axis=-1)

        self.running = True

    def _point_in_wall(self, point, wall_spec):
        """Check whether a point is inside a wall rectangle defined by wall_spec."""
        x, y = point
        wx, wy = wall_spec['x'], wall_spec['y']
        half_w = wall_spec['width'] / 2
        half_h = wall_spec['height'] / 2
        return (wx - half_w <= x <= wx + half_w) and (wy - half_h <= y <= wy + half_h)

    def _is_visible(self, start, end):
        """Check if the line from start to end is unobstructed by walls."""
        x0, y0 = start
        x1, y1 = end
        line = np.array(list(bresenham(x0, y0, x1, y1)))
        for x, y in line:
            if self.space.wall_grid[y, x]:
                return False
        return True

    def step(self):
        """Advance the model by one step."""
        # Update coverage based on robot positions
        for agent in self.agents:
            if isinstance(agent, RobotAgent):
                x, y = agent.pos
                grid_x, grid_y = self.space.continuous_to_grid(x, y)
                
                # Compute the absolute grid coordinates to update
                grid_positions = self.coverage_offsets + np.array([grid_x, grid_y])
                x_coords = grid_positions[:, 0]
                y_coords = grid_positions[:, 1]
                
                # Filter positions that fall outside the grid boundaries
                valid_mask = (
                    (x_coords >= 0) & (x_coords < self.grid_size) &
                    (y_coords >= 0) & (y_coords < self.grid_size)
                )
                x_coords = x_coords[valid_mask]
                y_coords = y_coords[valid_mask]
                
                # Further filter out cells that are walls by checking the wall grid
                not_wall = ~self.space.wall_grid[y_coords, x_coords]
                x_coords = x_coords[not_wall]
                y_coords = y_coords[not_wall]
                
                # Check line of sight for each valid position
                visible_mask = np.array([
                    self._is_visible((grid_x, grid_y), (x, y))
                    for x, y in zip(x_coords, y_coords)
                ])
                x_coords = x_coords[visible_mask]
                y_coords = y_coords[visible_mask]
                # --------------------------------------------
                
                # Mark these positions as covered
                self.coverage_grid[y_coords, x_coords] = True
        
        # Collect data
        self.datacollector.collect(self)
        
        # Step all agents
        for agent in self.agents:
            agent.step()
