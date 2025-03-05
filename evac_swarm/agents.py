import math
import random
import numpy as np
from mesa import Agent

# At the top of the file, define a global counter and helper function.
GLOBAL_AGENT_ID = 0

def get_next_agent_id():
    global GLOBAL_AGENT_ID
    next_id = GLOBAL_AGENT_ID
    GLOBAL_AGENT_ID += 1
    return next_id

# NEW: Helper function to reset the global agent counter.
def reset_agent_id_counter():
    global GLOBAL_AGENT_ID
    GLOBAL_AGENT_ID = 0
    
# A small helper function for distance calculation
def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

class RobotAgent(Agent):
    """
    A robot agent that can move in 360° directions, avoid collisions and detect casualties.
    """
    def __init__(self, model, vision_range=2.0, radius=0.3, move_behaviour="random", comm_timeout=10):
        Agent.__init__(self, model)
        self.unique_id = get_next_agent_id()
        self.orientation = model.random.uniform(0, 360)  # In degrees
        self.vision_range = vision_range
        self.radius = radius
        # A set to store aggregated casualty reports (e.g. positions)
        self.reported_casualties = set()

        self.move_speed = 0.3/2
        self.turn_speed = 8

        self.move_behaviour = move_behaviour
        
        # Communication parameters
        self.comm_timeout = comm_timeout  # How many steps before needing to communicate
        self.steps_since_comm = comm_timeout  # Start needing to communicate
        self.comm_range = vision_range  # Communication range (same as vision by default)
        
        # Personal coverage map (will be synced during communication)
        self.coverage = None  # Will be initialized in the model
        self.last_comm_partner = None  # Keep track of last agent communicated with

    def _attempt_move(self, distance):
        """Move the robot forward by a specified distance if no collision occurs."""
        rad = math.radians(self.orientation)
        new_x = self.pos[0] + distance * math.cos(rad)
        new_y = self.pos[1] + distance * math.sin(rad)
        new_pos = (new_x, new_y)

        # Check for collision
        if not self._detect_collision_fast(new_pos):
            # No collision, proceed with move
            self.model.space.move_agent(self, new_pos)
            self.pos = new_pos
            return True
        else:
            # Collision detected - try to adjust course
            self._adjust_course_for_collision()
            return False
            
    def _adjust_course_for_collision(self):
        """
        When a collision is detected, adjust the orientation to find a path around obstacles.
        Uses the _limit_turn function for realistic movement.
        """
        # Try different direction changes, respecting turn speed limits
        # Add a small random perturbation to avoid getting stuck in loops
        random_adjustment = self.model.random.uniform(-15, 15)
        self.orientation = self._limit_turn((self.orientation + random_adjustment) % 360)
        
    def _limit_turn(self, desired_angle):
        """
        Limit the change in angle from current_angle to desired_angle to at most max_change degrees.
        Angles are in degrees.
        """
        # Compute the smallest difference in the range [-180, 180]
        diff = (desired_angle - self.orientation + 180) % 360 - 180
        if diff > self.turn_speed:
            diff = self.turn_speed
        elif diff < -self.turn_speed:
            diff = -self.turn_speed
        # Return the updated orientation.
        return (self.orientation + diff) % 360

    def _disperse(self, neighbour_radius=3.0):
        """
        Adjust the agent's orientation to move away from nearby robots.

        Parameters:
            neighbour_radius: the distance within which neighbours affect the dispersion.
        """
        # Get nearby robot agents (excluding itself)
        neighbours = self.model.space.get_neighbors(self.pos, neighbour_radius, include_center=True)
        if not neighbours:
            return

        repulsion = np.array([0.0, 0.0])
        for nbr in neighbours:
            if isinstance(nbr, RobotAgent) and nbr.unique_id != self.unique_id:
                vector = np.array(self.pos) - np.array(nbr.pos)
                # If the robots are extremely close (or in the same spot), apply a small random nudge:
                if np.linalg.norm(vector) < 1e-5:
                    vector = np.array([self.model.random.uniform(-0.1, 0.1),
                                         self.model.random.uniform(-0.1, 0.1)])
                # Weight by inverse-square distance to emphasise closer neighbours
                distance = np.linalg.norm(vector) + 1e-6
                repulsion += vector / (distance ** 2)

        # If repulsion vector has magnitude, compute the desired angle.
        if np.linalg.norm(repulsion) > 0:
            desired_angle = np.degrees(np.arctan2(repulsion[1], repulsion[0]))
            # Limit the angle change to self.turn_speed (or another maximum)
            self.orientation = self._limit_turn(desired_angle)
            # Attempt to move with the new orientation at self.move_speed
            self._attempt_move(self.move_speed)
        else:
            self._random_exploration()

    def _random_exploration(self):
        """
        Execute a random (Brownian motion style) exploration.
        """
        # Generate a small random change in angle.
        delta_angle = self.model.random.gauss(0, self.turn_speed)
        self.orientation = (self.orientation + delta_angle) % 360

        # Attempt to move forward by self.move_speed if there is no collision.
        self._attempt_move(self.move_speed)

    def _find_communication_partner(self):
        """Find a nearby agent to communicate with, prioritising other robots"""
        # Filter for only robot and deployment agents
        communicable_agents = [agent for agent in self.model.agents 
                             if isinstance(agent, (RobotAgent, DeploymentAgent))]
        
        # Get nearby agents, passing in our filtered list
        nearby_agents, _ = self.model.space.get_agents_in_radius(
            agent=self, 
            radius=self.comm_range,
            agent_filter=communicable_agents
        )
        
        # First check for other robots
        for agent in nearby_agents:
            if isinstance(agent, RobotAgent) and agent.unique_id != self.unique_id:
                # Check if there's line of sight to this agent
                agent_pos = agent.pos
                grid_agent_x, grid_agent_y = self.model.space.continuous_to_grid(*agent_pos)
                grid_self_x, grid_self_y = self.model.space.continuous_to_grid(*self.pos)
                
                # Only proceed if we have line of sight (optional, can disable this check)
                if self.model.is_visible_vectorised(
                    (grid_self_x, grid_self_y),
                    (grid_agent_x, grid_agent_y),
                    self.model.space.wall_grid
                ):
                    return agent
        
        # If no robot is found, check for the deployment agent
        for agent in nearby_agents:
            if isinstance(agent, DeploymentAgent):
                # Check line of sight to deployment agent
                agent_pos = agent.pos
                grid_agent_x, grid_agent_y = self.model.space.continuous_to_grid(*agent_pos)
                grid_self_x, grid_self_y = self.model.space.continuous_to_grid(*self.pos)
                
                if self.model.is_visible_vectorised(
                    (grid_self_x, grid_self_y),
                    (grid_agent_x, grid_agent_y),
                    self.model.space.wall_grid
                ):
                    return agent
        
        return None
    
    def communicate(self, partner):
        """Share coverage information with another agent"""
        # Only perform data exchange if the partner is a robot
        if isinstance(partner, RobotAgent):
            # Combine coverage maps (logical OR)
            if self.coverage is not None and partner.coverage is not None:
                combined_coverage = np.logical_or(self.coverage, partner.coverage)
                self.coverage = combined_coverage.copy()
                partner.coverage = combined_coverage.copy()
            
            # Share casualty reports
            self.reported_casualties.update(partner.reported_casualties)
            partner.reported_casualties.update(self.reported_casualties)
            
            # Reset communication timer for both agents
            self.steps_since_comm = 0
            partner.steps_since_comm = 0
            
            # Update last communication partner
            self.last_comm_partner = partner.unique_id
            partner.last_comm_partner = self.unique_id
            
            return True
        
        # If communicating with deployment agent, bidirectional information exchange
        elif isinstance(partner, DeploymentAgent):
            # Exchange coverage information (bidirectional)
            if self.coverage is not None and partner.global_coverage is not None:
                # Merge robot's coverage into deployment agent's global coverage
                partner.global_coverage = np.logical_or(partner.global_coverage, self.coverage)
                # Also get updated coverage from deployment agent
                self.coverage = np.logical_or(self.coverage, partner.global_coverage)
            
            # Exchange casualty reports
            self.reported_casualties.update(partner.reported_casualties)
            partner.reported_casualties.update(self.reported_casualties)
            
            # Add robot to reported list
            partner.robots_reported.add(self.unique_id)
            
            # Reset communication timer
            self.steps_since_comm = 0
            self.last_comm_partner = partner.unique_id

            return True
            
        return False
    
    def _seek_communication(self):
        """Behavior to seek out communication with other agents"""
        # Find potential communication partners
        partner = self._find_communication_partner()
        
        if partner:
            # If partner found, communicate and don't move this step
            self.communicate(partner)
            return True
        
        # No partner in range, continue with random exploration
        self._random_exploration()
        return False

    def _update_coverage(self):
        """Update the agent's personal coverage map based on current position"""
        if self.coverage is None:
            # Initialize personal coverage grid if not already done
            self.coverage = np.zeros((self.model.space.num_cells_y, self.model.space.num_cells_x), dtype=bool)
        
        # Update personal coverage based on current position
        x, y = self.pos
        grid_x, grid_y = self.model.space.continuous_to_grid(x, y)
        
        # Apply coverage offsets (similar to the model's coverage update)
        grid_positions = self.model.coverage_offsets + np.array([grid_x, grid_y])
        x_coords = np.clip(grid_positions[:, 0], 0, self.model.space.num_cells_x - 1)
        y_coords = np.clip(grid_positions[:, 1], 0, self.model.space.num_cells_y - 1)
        
        # Filter out wall positions
        not_wall = ~self.model.space.wall_grid[y_coords, x_coords]
        x_coords = x_coords[not_wall]
        y_coords = y_coords[not_wall]
        
        # Mark visible areas in personal coverage
        self.coverage[y_coords, x_coords] = True

    def _detect_collision_fast(self, new_pos):
        """Check collisions with walls using the model's spatial index for accelerated lookup.
        Returns True if a collision is detected, False otherwise.
        """
        x, y = new_pos
        # Check simulation bounds.
        if x < 0 or x > self.model.width or y < 0 or y > self.model.height:
            return True

        # Define a bounding box around the new position.
        bbox = (x - self.radius, y - self.radius, x + self.radius, y + self.radius)

        # Query the spatial index for candidate wall agents.
        for candidate in self.model.wall_index.intersection(bbox, objects=True):
            wall_spec = candidate.object
            if RobotAgent.collides_with_wall(new_pos, self.radius, wall_spec):
                return True
        return False

    @staticmethod
    def collides_with_wall(new_pos, radius, wall_spec):
        """
        Collision detection for a circular agent (with radius)
        and a rectangular wall defined by wall_spec.
        Returns True if the distance from new_pos to the wall is less than radius.
        """
        # TODO: Implement collision detection using the R-tree spatial index in model.distance_map.
        
        px, py = new_pos
        wx, wy = wall_spec['x'], wall_spec['y']
        half_w = wall_spec['width'] / 2
        half_h = wall_spec['height'] / 2

        # Find the closest point on the rectangle to the circle centre.
        closest_x = max(wx - half_w, min(px, wx + half_w))
        closest_y = max(wy - half_h, min(py, wy + half_h))

        # Calculate the Euclidean distance.
        distance = math.hypot(px - closest_x, py - closest_y)
        return distance < radius

    def _detect_casualties(self):
        """Detect casualty agents within vision range and update the report."""
        for agent in self.model.agents:
            if type(agent).__name__ == "CasualtyAgent":
                if euclidean_distance(self.pos, agent.pos) <= self.vision_range:
                    # Report casualty by adding its position
                    self.reported_casualties.add(agent.pos)
                    agent.discovered = True

    def step(self):
        # Increment step counter since last communication
        self.steps_since_comm += 1
        
        # Decide on behavior based on communication needs
        if self.steps_since_comm >= self.comm_timeout:
            # Need to communicate - prioritize finding a partner
            communicated = self._seek_communication()

        # Normal exploration behavior
        if self.move_behaviour == "disperse":
            self._disperse()
        else:
            self._random_exploration()
        
        # After moving, update coverage and detect casualties
        self._update_coverage()
        self._detect_casualties()


class WallAgent(Agent):
    """
    A wall agent representing an impassable structure.
    """
    def __init__(self, model, wall_spec):
        Agent.__init__(self, model)
        self.unique_id = get_next_agent_id()
        self.wall_spec = wall_spec  # a dict with x, y, width, height
        
    def step(self):
        # Walls are static so no behaviour on step.
        pass
    
    
class CasualtyAgent(Agent):
    """
    A casualty agent. Static until detected.
    """
    def __init__(self, model):
        Agent.__init__(self, model)
        self.unique_id = get_next_agent_id()
        self.discovered = False
    
    def step(self):
        # Casualties do not move.
        pass

class DeploymentAgent(Agent):
    """
    A deployment agent that acts as a communication hub for robot agents.
    """
    def __init__(self, model):
        Agent.__init__(self, model)
        self.unique_id = get_next_agent_id()
        # Store a global coverage map that represents the combined knowledge of all robots
        # that have communicated with the deployment agent
        self.global_coverage = None
        # Track which robots have reported back
        self.robots_reported = set()
        # Store all casualty locations reported by robots
        self.reported_casualties = set()

    def initialize_coverage_map(self, shape):
        """Initialize the global coverage map once grid dimensions are known"""
        if self.global_coverage is None:
            self.global_coverage = np.zeros(shape, dtype=bool)
    
    def update_from_robot(self, robot):
        """Update global knowledge from a robot's report"""
        # Add robot to reported list
        self.robots_reported.add(robot.unique_id)
        
        # Update global coverage map with robot's personal coverage
        if robot.coverage is not None and self.global_coverage is not None:
            self.global_coverage = np.logical_or(self.global_coverage, robot.coverage)
        
        # Update casualty reports
        self.reported_casualties.update(robot.reported_casualties)
        
        return True

    def step(self):
        # Deployment agent does not move or contribute to coverage.
        # Communication happens when robots initiate it.
        pass 