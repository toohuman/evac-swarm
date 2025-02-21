import random
import numpy as np
from mesa import Model
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector
from mesa.experimental.cell_space import PropertyLayer
from mesa.experimental.devs import ABMSimulator
from scipy.ndimage import binary_dilation, distance_transform_edt
from rtree import index

from evac_swarm.agents import RobotAgent, WallAgent, CasualtyAgent, DeploymentAgent
from evac_swarm.building_generator import generate_building_layout
from evac_swarm.space import HybridSpace

# Delete if left unused
def bresenham(x0, y0, x1, y1):
    """
    Bresenham's Line Algorithm.
    Yields integer coordinates on the line from (x0, y0) to (x1, y1).
    """
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy  # error value e_xy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy

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
        move_behaviour="random",
        grid_size=100,
        show_vision_range=True,
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
        self.move_behaviour = move_behaviour
        self.grid_size = grid_size
        self.show_vision_range = show_vision_range
        
        self.simulator = simulator
        if self.simulator is not None:
            self.simulator.setup(self)  # Ensure the simulator is set up on the model instance.
        
        # Determine the aspect ratio
        aspect_ratio = max(self.width, self.height) / min(self.width, self.height)

        # Adjust grid size to maintain square cells
        self.grid_size = int(self.grid_size * aspect_ratio)

        # Calculate grid dimensions (height × width)
        num_cells_y = int(self.grid_size * (self.height / max(self.width, self.height)))
        num_cells_x = int(self.grid_size * (self.width / max(self.width, self.height)))

        
        # Initialize both grids consistently as (rows, cols) or (y, x)
        self.coverage_grid = np.zeros((num_cells_y, num_cells_x), dtype=np.int8)  # [y, x] (0 = not covered, 1 = covered, -1 = wall)
        self.space = HybridSpace(self.width, self.height, 
                               dims=(num_cells_y, num_cells_x), torus=False)  # Pass as (y, x)
        
        # Generate building layout with new seed
        wall_layout = generate_building_layout(
            self.width, self.height, 
            self.min_room_size, 
            self.wall_thickness,
            rng=self.random  # Use new seed for building
        )
        
        self._next_id = 0  # Add counter for agent IDs
        
        # Add walls to both representations
        for wall in wall_layout:
            self.space.add_wall(wall)
            # Create a WallAgent for each wall
            wall_agent = WallAgent(self._next_id, self, wall_spec=wall)
            self._next_id += 1
            self.register_agent(wall_agent)
        
        # Mark wall cells in the coverage grid with -1 so they are not counted as accessible.
        self.coverage_grid[self.space.wall_grid] = -1
        
        # Recalculate total accessible cells (non-wall cells) AFTER walls have been added.
        self.total_accessible_cells = self.coverage_grid.size - np.sum(self.space.wall_grid)

        # Update DataCollector
        self.datacollector = DataCollector(
            model_reporters={
                "Coverage": lambda m: (np.sum(m.coverage_grid == 1) / m.total_accessible_cells) * 100
            }
        )
        
        # Build an R-tree spatial index for the wall agents.
        # Assume wall_agent.unique_id is unique.
        self.wall_index = index.Index()
        for agent in self.agents:
            if isinstance(agent, WallAgent):
                wx, wy = agent.wall_spec['x'], agent.wall_spec['y']
                half_w = agent.wall_spec['width'] / 2.0
                half_h = agent.wall_spec['height'] / 2.0
                bbox = (wx - half_w, wy - half_h, wx + half_w, wy + half_h)
                # Instead of storing the agent (which cannot be pickled), store just the wall_spec.
                self.wall_index.insert(agent.unique_id, bbox, obj=agent.wall_spec)
        
        # Precompute the distance map for collision detection.
        # Compute the Euclidean distance (in grid cells) from each free cell to the nearest wall.
        # Note: wall_grid is True for walls, so invert it.
        self.distance_map = distance_transform_edt(~self.space.wall_grid)
        
        # Define the entry point (deployment operator location).
        # We assume the entry is at the centre of the bottom wall.
        self.entry_point = (self.width / 2, 1 + self.wall_thickness)
        
        # Add a DeploymentAgent at the entry point.
        deployment_agent = DeploymentAgent(self._next_id, self, self.entry_point)
        self._next_id += 1
        self.register_agent(deployment_agent)
        self.space.place_agent(deployment_agent, self.entry_point)
        
        # Place Robot agents at the entry point.
        for _ in range(self.robot_count):
            robot = RobotAgent(
                self._next_id,
                self,  # model
                vision_range=self.vision_range,
                move_behaviour=self.move_behaviour
            )
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
        # Use np.ceil to ensure that the discrete grid vision range fully covers the continuous vision range.
        vision_grid_range = int(np.ceil(self.vision_range * min(self.space.num_cells_x / self.width, 
                                                                self.space.num_cells_y / self.height)))
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

    # def _is_visible(self, start, end):
    #     """Check if the line from start to end is unobstructed by walls."""
    #     x0, y0 = start
    #     x1, y1 = end
    #     line = np.array(list(bresenham(x0, y0, x1, y1)))
    #     for x, y in line:
    #         if self.space.wall_grid[y, x]:
    #             return False
    #     return True
    
    def _is_visible_vectorized(self, start, end, wall_grid):
        """
        Determines if the line from start to end is clear of walls.
        
        Parameters:
            start: (x, y) tuple representing the start coordinates in grid space.
            end: (x, y) tuple representing the end coordinates in grid space.
            wall_grid: A 2D boolean NumPy array where True indicates a wall.
        
        Returns:
            True if there is a clear line from start to end, False if any cell is blocked.
        """
        # Compute Euclidean distance from start to end:
        distance = np.hypot(end[0] - start[0], end[1] - start[1])
        # Number of sample points along the line.
        num_points = int(np.ceil(distance)) + 1
        
        # Generate linearly spaced coordinates along the line:
        xs = np.linspace(start[0], end[0], num_points)
        ys = np.linspace(start[1], end[1], num_points)
        
        # Convert these coordinates to grid indices:
        xs_int = np.clip(np.round(xs).astype(int), 0, wall_grid.shape[1] - 1)
        ys_int = np.clip(np.round(ys).astype(int), 0, wall_grid.shape[0] - 1)
        
        # Index the wall grid vectorised:
        cells_along_line = wall_grid[ys_int, xs_int]
        
        # If any cell is a wall, line-of-sight is obstructed.
        return not np.any(cells_along_line)


    def step(self):
        """Advance the model by one step using vectorized updates for robots."""
        for agent in self.agents:
            if isinstance(agent, RobotAgent):
                x, y = agent.pos
                grid_x, grid_y = self.space.continuous_to_grid(x, y)
                
                # Compute the absolute grid coordinates to update
                grid_positions = self.coverage_offsets + np.array([grid_x, grid_y])
                x_coords = np.clip(grid_positions[:, 0], 0, self.space.num_cells_x - 1)
                y_coords = np.clip(grid_positions[:, 1], 0, self.space.num_cells_y - 1)

                # Access both grids consistently with [y, x]
                not_wall = ~self.space.wall_grid[y_coords, x_coords]
                x_coords = x_coords[not_wall]
                y_coords = y_coords[not_wall]

                # Check line of sight for each valid position
                # visible_mask = np.array([
                #     self._is_visible_vectorized((grid_x, grid_y), (x, y), self.space.wall_grid)
                #     for x, y in zip(x_coords, y_coords)
                # ])
                # x_coords = x_coords[visible_mask]
                # y_coords = y_coords[visible_mask]
                # --------------------------------------------

                # Mark coverage consistently with [y, x]
                self.coverage_grid[y_coords, x_coords] = 1

        # Step any remaining agents that don't get handled in the vectorized update.
        for agent in self.agents:
            agent.step()

        # Collect data after updates.
        self.datacollector.collect(self)
