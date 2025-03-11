import math
import random
import numpy as np
from mesa import Agent
from typing import List, Set, Tuple, Optional
from evac_swarm.communication import Message, CoverageMessage, CasualtyMessage


def euclidean_distance(a, b):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


class RobotAgent(Agent):
    """
    A robot agent that can move in 360° directions, avoid collisions and detect casualties.
    """
    def __init__(self, model, vision_range=2.0, radius=0.3, move_behaviour="random", comm_timeout=10):
        """
        Initialize a robot agent.
        
        Args:
            model: The model instance
            vision_range: How far the robot can see
            radius: Physical size of the robot
            move_behaviour: Movement strategy ("random", "disperse", etc.)
            comm_timeout: Steps before seeking communication
        """
        # Use the Mesa Agent initialization pattern
        super().__init__(model)
        
        self.orientation = model.random.uniform(0, 360)  # In degrees
        self.vision_range = vision_range
        self.radius = radius  # Physical size of the robot
        self.move_speed = 0.5  # How far the robot moves in one step
        self.turn_speed = 30  # Maximum degrees to turn in one step
        self.move_behaviour = move_behaviour  # Movement strategy
        
        # Communication parameters
        self.steps_since_comm = 0
        self.comm_timeout = comm_timeout
        self.comm_range = 5.0  # Range for communication
        self.pending_messages: List[Message] = []  # Messages waiting to be sent
        
        # A set to store aggregated casualty reports (e.g. positions)
        self.reported_casualties = set()
        
        # Personal coverage map (will be synced during communication)
        self.coverage = None  # Will be initialized by the model
        self.last_comm_partner = None

    def _attempt_move(self, distance):
        """Move the robot forward by a specified distance if no collision occurs."""
        rad = math.radians(self.orientation)
        new_x = self.pos[0] + distance * math.cos(rad)
        new_y = self.pos[1] + distance * math.sin(rad)
        new_pos = (new_x, new_y)

        # Check for collision
        if not self._detect_collision(new_pos):
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

    def _disperse(self, neighbour_radius=None):
        """
        Adjust the agent's orientation to move away from nearby robots.

        Parameters:
            neighbour_radius: the distance within which neighbours affect the dispersion.
        """
        if neighbour_radius is None:
            neighbour_radius = self.vision_range

        # Filter for only robot agents
        robot_agents = [agent for agent in self.model.agents 
                      if isinstance(agent, RobotAgent)]
        
        # Get nearby robot agents using get_agents_in_radius
        try:
            neighbours, _ = self.model.space.get_agents_in_radius(
                agent=self,
                radius=neighbour_radius,
                agent_filter=robot_agents
            )
        except Exception as e:
            print(f"Error finding neighbours for dispersion: {e}")
            return

        repulsion = np.array([0.0, 0.0])
        for nbr in neighbours:
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
        """
        Find a suitable communication partner within range.
        
        Returns:
            Agent or None: A communication partner if found, None otherwise
        """
        # Filter for only robot and deployment agents
        communicable_agents = [agent for agent in self.model.agents 
                             if isinstance(agent, (RobotAgent, DeploymentAgent))]
        
        # Get nearby agents, passing in our filtered list
        try:
            nearby_agents, _ = self.model.space.get_agents_in_radius(
                agent=self, 
                radius=self.comm_range,
                agent_filter=communicable_agents
            )
        except Exception as e:
            print(f"Error finding communication partners: {e}")
            return None
            
        if not nearby_agents:
            return None
        
        # Generate random indices to check agents in random order
        random_indices = self.model.random.sample(range(len(nearby_agents)), len(nearby_agents))
        
        # Check agents in random order
        for idx in random_indices:
            agent = nearby_agents[idx]
            if agent.unique_id != self.unique_id:
                # Check if there's line of sight to this agent
                try:
                    agent_pos = agent.pos
                    grid_agent_x, grid_agent_y = self.model.space.continuous_to_grid(*agent_pos)
                    grid_self_x, grid_self_y = self.model.space.continuous_to_grid(*self.pos)
                    
                    # Only proceed if we have line of sight
                    if self.model.is_visible_vectorised(
                        (grid_self_x, grid_self_y),
                        (grid_agent_x, grid_agent_y),
                        self.model.space.wall_grid
                    ):
                        return agent
                except Exception as e:
                    print(f"Error checking line of sight to agent: {e}")
                    continue
        
        return None

    def prepare_coverage_message(self):
        """
        Prepare a message containing this robot's coverage data.
        """
        if self.coverage is not None:
            message = self.model.communication_manager.create_message(
                sender=self,
                message_type="coverage",
                coverage_data=self.coverage
            )
            self.pending_messages.append(message)

    def prepare_casualty_message(self):
        """
        Prepare a message containing this robot's casualty reports.
        """
        if self.reported_casualties:
            message = self.model.communication_manager.create_message(
                sender=self,
                message_type="casualty",
                casualty_positions=self.reported_casualties
            )
            self.pending_messages.append(message)

    def send_messages(self):
        """
        Send all pending messages to nearby agents.
        
        Returns:
            bool: True if messages were successfully sent, False otherwise
        """
        if not self.pending_messages:
            return False
            
        try:
            # Filter for only communicable agents with valid positions
            communicable_agents = [
                agent for agent in self.model.agents 
                if isinstance(agent, (RobotAgent, DeploymentAgent))
            ]
            
            recipients = self.model.communication_manager.deliver_messages(
                sender=self,
                messages=self.pending_messages,
                comm_range=self.comm_range
            )
            
            # Reset communication timer if we successfully communicated
            if recipients:
                self.steps_since_comm = 0
                
            # Clear pending messages
            self.pending_messages = []
            
            return bool(recipients)
        except Exception as e:
            print(f"Error sending messages from robot agent: {e}")
            # Clear pending messages to avoid repeated errors
            self.pending_messages = []
            return False

    def communicate(self, partner) -> None:
        """
        Communicate with other agents via the message system.
        
        Args:
            partner: The agent to communicate with
        """
        # Prepare messages
        self.prepare_coverage_message()
        self.prepare_casualty_message()
        
        # Send messages directly to partner
        if self.pending_messages:
            self.model.communication_manager.deliver_to_agent(partner, self.pending_messages)
            self.pending_messages = []
            self.steps_since_comm = 0

    def _seek_communication(self):
        """
        Actively seek a communication partner.
        
        Returns:
            bool: True if communication occurred, False otherwise
        """
        # Find a communication partner
        partner = self._find_communication_partner()
        
        if partner:
            # Use the legacy communication method for now
            self.communicate(partner)
            return True
            
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

    def _detect_collision(self, new_pos):
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
        """
        Execute one step of the agent's behavior.
        """
        # Increment step counter since last communication
        self.steps_since_comm += 1
        
        # Update coverage map based on current position
        self._update_coverage()
        
        # Check for casualties in vision range
        self._detect_casualties()
        
        # Decide on behavior based on communication needs
        if self.steps_since_comm >= self.comm_timeout:
            # Need to communicate - prioritize finding a partner
            communicated = self._seek_communication()
            
        # Regular exploration
        if self.move_behaviour == "disperse":
            self._disperse()
        else:
            self._random_exploration()
                
        # Try to send any pending messages
        self.send_messages()


class WallAgent(Agent):
    """
    A wall agent that represents a physical barrier in the environment.
    """

    def __init__(self, model, wall_spec):
        super().__init__(model)
        self.wall_spec = wall_spec

    def step(self):
        # Walls are static so no behaviour on step.
        pass


class CasualtyAgent(Agent):
    """
    A casualty agent that represents a person in need of rescue.
    """

    def __init__(self, model):
        super().__init__(model)
        self.discovered = False

    def step(self):
        # Casualties do not move.
        pass

class DeploymentAgent(Agent):
    """
    Represents the deployment point for robots. Acts as a communication hub.
    """

    def __init__(self, model):
        """
        Initialize a deployment agent.
        
        Args:
            model: The model instance
        """
        super().__init__(model)
        self.global_coverage = None  # Will be initialized later
        self.reported_casualties = set()  # Set of casualty positions
        self.robots_reported = set()  # Set of robot IDs that have reported
        self.pending_messages: List[Message] = []  # Messages waiting to be sent
        self.comm_range = 10.0  # Larger communication range than robots

    def initialize_coverage_map(self, shape):
        """Initialize the global coverage map with the given shape"""
        self.global_coverage = np.zeros(shape, dtype=bool)

    def prepare_coverage_message(self):
        """
        Prepare a message containing the global coverage data.
        """
        if self.global_coverage is not None:
            message = self.model.communication_manager.create_message(
                sender=self,
                message_type="coverage",
                coverage_data=self.global_coverage
            )
            self.pending_messages.append(message)

    def prepare_casualty_message(self):
        """
        Prepare a message containing all reported casualties.
        """
        if self.reported_casualties:
            message = self.model.communication_manager.create_message(
                sender=self,
                message_type="casualty",
                casualty_positions=self.reported_casualties
            )
            self.pending_messages.append(message)

    def send_messages(self):
        """
        Send all pending messages to nearby agents.
        """
        if not self.pending_messages:
            return
            
        try:
            # Filter for only robot agents with valid positions
            communicable_agents = [
                agent for agent in self.model.agents 
                if isinstance(agent, RobotAgent)
            ]
            
            self.model.communication_manager.deliver_messages(
                sender=self,
                messages=self.pending_messages,
                comm_range=self.comm_range
            )
            self.pending_messages = []
        except Exception as e:
            print(f"Error sending messages from deployment agent: {e}")
            # Clear pending messages to avoid repeated errors
            self.pending_messages = []

    def step(self):
        """
        Execute one step of the deployment agent's behavior.
        """
        # Deployment agent does not move
        # It will respond to communication initiated by robots
        # But can also proactively send messages to nearby robots
        self.prepare_coverage_message()
        self.prepare_casualty_message()
        self.send_messages() 