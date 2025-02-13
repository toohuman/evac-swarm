from mesa.space import ContinuousSpace
import numpy as np

class HybridSpace(ContinuousSpace):
    """A hybrid space that combines continuous agent movement with a discrete grid for walls."""
    
    def __init__(self, x_max, y_max, dims=(100, 100), torus=False):
        """
        Args:
            x_max, y_max: Continuous space dimensions
            grid_size: Number of grid cells for discrete features
            torus: Whether the space wraps around
        """
        super().__init__(x_max, y_max, torus)
        self.num_cells_y, self.num_cells_x = dims  # Unpack as (y, x)
        self.wall_grid = np.zeros((self.num_cells_y, self.num_cells_x), dtype=bool)  # Initialize as [y, x]
        self.wall_specs = []  # Store original wall specifications
        
    def add_wall(self, wall_spec):
        """Add a wall to both representations."""
        self.wall_specs.append(wall_spec)
        
        # Update grid representation
        x, y = wall_spec['x'], wall_spec['y']
        width, height = wall_spec['width'], wall_spec['height']
        
        # Convert wall rectangle to grid cells
        min_x = max(0, int((x - width/2) * self.num_cells_x / self.x_max))
        max_x = min(self.num_cells_x, int((x + width/2) * self.num_cells_x / self.x_max))
        min_y = max(0, int((y - height/2) * self.num_cells_y / self.y_max))
        max_y = min(self.num_cells_y, int((y + height/2) * self.num_cells_y / self.y_max))
        
        self.wall_grid[min_y:max_y, min_x:max_x] = True
        
    def continuous_to_grid(self, x, y):
        """Convert continuous coordinates to grid coordinates."""
        grid_x = int((x / self.x_max) * self.num_cells_x)
        grid_y = int((y / self.y_max) * self.num_cells_y)
        return grid_x, grid_y
    
    def grid_to_continuous(self, grid_x, grid_y):
        """Convert grid coordinates to continuous coordinates."""
        x = (grid_x / self.num_cells_x) * self.x_max
        y = (grid_y / self.num_cells_y) * self.y_max
        return x, y
    
    def is_wall_at(self, x, y):
        """Check if there's a wall at the given continuous coordinates."""
        grid_x, grid_y = self.continuous_to_grid(x, y)
        try:
            return bool(self.wall_grid[grid_y, grid_x])
        except IndexError:
            return False 