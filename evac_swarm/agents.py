import math
import random
import numpy as np
from mesa import Agent

# A small helper function for distance calculation
def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

class RobotAgent(Agent):
    """
    A robot agent that can move in 360° directions, avoid collisions and detect casualties.
    """
    def __init__(self, unique_id, model, vision_range=2.0, radius=0.3, move_behaviour="random"):
        Agent.__init__(self, model)
        self.unique_id = unique_id
        self.orientation = self.model.random.uniform(0, 360)  # In degrees
        self.vision_range = vision_range
        self.radius = radius
        # A set to store aggregated casualty reports (e.g. positions)
        self.reported_casualties = set()

        self.move_speed = 0.3/2
        self.turn_speed = 8

        self.move_behaviour = move_behaviour

    def attempt_move(self, distance):
        """Move the robot forward by a specified distance if no collision occurs."""
        rad = math.radians(self.orientation)
        new_x = self.pos[0] + distance * math.cos(rad)
        new_y = self.pos[1] + distance * math.sin(rad)
        new_pos = (new_x, new_y)

        if not self.detect_collision_fast(new_pos):
            self.model.space.move_agent(self, new_pos)
            self.pos = new_pos

    def limit_turn(self, desired_angle):
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

    def disperse(self, neighbour_radius=3.0):
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
            self.orientation = self.limit_turn(desired_angle)
            # Attempt to move with the new orientation at self.move_speed
            self.attempt_move(self.move_speed)
        else:
            self.random_exploration()

    def random_exploration(self):
        """
        Execute a random (Brownian motion style) exploration.
        """
        # Generate a small random change in angle.
        delta_angle = self.model.random.gauss(0, self.turn_speed)
        self.orientation = (self.orientation + delta_angle) % 360

        # Attempt to move forward by self.move_speed if there is no collision.
        self.attempt_move(self.move_speed)

    def step(self):
        # Choose the movement behavior based on the current mode.
        if self.move_behaviour == "disperse":
            self.disperse()
        else:
            self.random_exploration()

        # After moving, attempt to detect casualties.
        self.detect_casualties()

    def detect_collision(self, new_pos):
        """Check collisions with walls and other robot agents."""
        # Check for wall collisions
        for agent in self.model.agents:
            if isinstance(agent, WallAgent):
                if self.collides_with_wall(new_pos, self.radius, agent.wall_spec):
                    return True
                # elif type(agent).__name__ == "RobotAgent" and agent.unique_id != self.unique_id:
                #     if euclidean_distance(new_pos, agent.pos) < (self.radius * 2):
                #         return True

        # Check if the new position is out of bounds
        x, y = new_pos
        if x < 0 or x > self.model.width or y < 0 or y > self.model.height:
            return True

        return False

    def detect_collision_fast(self, new_pos):
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

    def detect_casualties(self):
        """Detect casualty agents within vision range and update the report."""
        for agent in self.model.agents:
            if type(agent).__name__ == "CasualtyAgent":
                if euclidean_distance(self.pos, agent.pos) <= self.vision_range:
                    # Report casualty by adding its position
                    self.reported_casualties.add(agent.pos)
                    agent.discovered = True

class WallAgent(Agent):
    """
    A wall agent representing an impassable structure.
    """
    def __init__(self, unique_id, model, wall_spec):
        Agent.__init__(self, model)
        self.unique_id = unique_id
        self.wall_spec = wall_spec  # a dict with x, y, width, height
        
    def step(self):
        # Walls are static so no behaviour on step.
        pass
    
    
class CasualtyAgent(Agent):
    """
    A casualty agent. Static until detected.
    """
    def __init__(self, unique_id, model, pos):
        Agent.__init__(self, model)
        self.unique_id = unique_id
        self.discovered = False
    
    def step(self):
        # Casualties do not move.
        pass

class DeploymentAgent(Agent):
    """
    A deployment agent that acts as a communication hub for robot agents.
    """
    def __init__(self, unique_id, model, pos):
        Agent.__init__(self, model)
        self.unique_id = unique_id
        self.pos = pos

    def step(self):
        # Deployment agent does not move or contribute to coverage.
        # It can communicate with robot agents if needed.
        pass 