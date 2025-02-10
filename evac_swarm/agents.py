import math
import random
from mesa import Agent

# A small helper function for distance calculation
def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

class RobotAgent(Agent):
    """
    A robot agent that can move in 360° directions, avoid collisions and detect casualties.
    """
    def __init__(self, unique_id, model, pos, vision_range, radius=0.3):
        Agent.__init__(self, model)
        self.unique_id = unique_id
        self.orientation = self.model.random.uniform(0, 360)  # In degrees
        self.vision_range = vision_range
        self.radius = radius
        # A set to store aggregated casualty reports (e.g. positions)
        self.reported_casualties = set()

        self.move_speed = 0.3/2
        self.turn_speed = 8

    def move_forward(self, distance):
        """Move the robot forward by a specified distance."""
        rad = math.radians(self.orientation)
        new_x = self.pos[0] + distance * math.cos(rad)
        new_y = self.pos[1] + distance * math.sin(rad)
        new_pos = (new_x, new_y)
        
        if not self.detect_collision_fast(new_pos):
            self.model.space.move_agent(self, new_pos)
            self.pos = new_pos

    def turn_left(self, angle):
        """Turn the robot left by a specified angle."""
        self.orientation = (self.orientation - angle) % 360

    def turn_right(self, angle):
        """Turn the robot right by a specified angle."""
        self.orientation = (self.orientation + angle) % 360

    def step(self):
        # Brownian motion style update:
        # Instead of choosing a fixed left/right turn, update the orientation with
        # a small random change sampled from a normal distribution.
        delta_angle = self.model.random.gauss(0, self.turn_speed)
        self.orientation = (self.orientation + delta_angle) % 360

        # Move forward by a fixed step length.
        self.move_forward(self.move_speed)
        
        # Attempt to detect casualties within vision_range.
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

    def move(self, new_position):
        """Move to new position if there's no wall there."""
        x, y = new_position
        if not self.model.space.is_wall_at(x, y):
            self.model.space.move_agent(self, new_position)

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