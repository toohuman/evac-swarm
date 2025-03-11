from collections.abc import Iterable
from itertools import compress
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist

from mesa.agent import Agent, AgentSet
from mesa.space import ContinuousSpace

class HybridSpace(ContinuousSpace):
    """A hybrid space that combines continuous agent movement with a discrete grid for walls."""
    
    def __init__(self, model, x_max: float, y_max: float, dims: Tuple[int, int] = (100, 100), 
                 torus: bool = False, n_agents: int = 100) -> None:
        """
        Args:
            x_max, y_max: Continuous space dimensions
            grid_size: Number of grid cells for discrete features
            torus: Whether the space wraps around
        """
        super().__init__(x_max, y_max, torus)
        self.model = model
        self.num_cells_y, self.num_cells_x = dims  # Unpack as (y, x)
        self.wall_grid = np.zeros((self.num_cells_y, self.num_cells_x), dtype=bool)  # Initialize as [y, x]
        self.wall_specs = []  # Store original wall specifications

        self.last_step = -1

    def calculate_agent_distances(self, agent_list: list[Agent]) -> None:
        """Calculate pairwise distances between all agents using vectorised operations.
        
        This function creates a distance matrix storing distances between all agents,
        which can be used later for efficient neighbour lookups. The matrix is stored
        as a class variable agent_distance_matrix.
        """
        # Ensure all positions are 2D numpy arrays with shape (N, 2)
        positions = []
        agent_ids = []  # Store unique IDs for each agent
        for agent in agent_list:
            pos = agent.pos
            # Convert position to numpy array if it isn't already
            if not isinstance(pos, np.ndarray):
                pos = np.array(pos)
            # Ensure it's a 1D array with 2 elements (x, y)
            pos = pos.flatten()[:2]
            positions.append(pos)
            agent_ids.append(agent.unique_id)
            
        # Stack all positions into a single 2D array
        positions = np.vstack(positions)
        
        # Calculate pairwise distances between all agents
        self.agent_distance_matrix = cdist(positions, positions)
        
        # Store agent IDs to map matrix indices to agents
        self.agent_ids = agent_ids
        
        # Store the step number when this was last calculated
        self.last_step = self.model.steps

    def get_agents_in_radius(self, agent: Agent, radius: float, agent_filter: list[Agent] = None) -> tuple[list, np.ndarray]:
        """Get all agents within a specified radius of the given agent.
        
        Args:
            agent: The agent to find neighbours for
            radius: The radius within which to find neighbours
            agent_filter: Optional list of agents to consider. If None, uses all agents.
            
        Returns:
            tuple: (list of agents within radius, their distances from the agent)
        """
        # If we're in a new step or agent_filter has changed, recalculate distances
        if self.model.steps != self.last_step:
            # Use filtered agents if provided, otherwise use all agents
            agents_to_consider = agent_filter if agent_filter is not None else list(self.model.agents)
            self.calculate_agent_distances(agents_to_consider)
            
        # Get the index of the current agent in our stored list using its unique ID
        agent_idx = self.agent_ids.index(agent.unique_id)
        
        # Get distances to all other agents from the distance matrix
        distances = self.agent_distance_matrix[agent_idx]
        
        # Filter agents by radius
        logical = distances <= radius
        
        # Get the filtered agent IDs and distances
        filtered_ids = [aid for aid, mask in zip(self.agent_ids, logical) if mask]
        filtered_distances = distances[logical]
        
        # Remove the agent itself from the results
        if agent.unique_id in filtered_ids:
            agent_self_idx = filtered_ids.index(agent.unique_id)
            filtered_ids.pop(agent_self_idx)
            filtered_distances = np.delete(filtered_distances, agent_self_idx)
            
        # Convert IDs back to agent references using the model's agent lookup
        agents = [a for a in self.model.agents if a.unique_id in filtered_ids]
            
        return (agents, filtered_distances)
        
    def add_wall(self, wall_spec: dict) -> None:
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
        
    def continuous_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert continuous coordinates to grid coordinates."""
        grid_x = int((x / self.x_max) * self.num_cells_x)
        grid_y = int((y / self.y_max) * self.num_cells_y)
        return grid_x, grid_y
    
    def grid_to_continuous(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to continuous coordinates."""
        x = (grid_x / self.num_cells_x) * self.x_max
        y = (grid_y / self.num_cells_y) * self.y_max
        return x, y
    
    def is_wall_at(self, x: float, y: float) -> bool:
        """Check if there's a wall at the given continuous coordinates."""
        grid_x, grid_y = self.continuous_to_grid(x, y)
        try:
            return bool(self.wall_grid[grid_y, grid_x])
        except IndexError:
            return False 