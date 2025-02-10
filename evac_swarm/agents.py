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

        self.move_speed = 0.5/60
        self.turn_speed = 3

    def move_forward(self, distance):
        """Move the robot forward by a specified distance."""
        rad = math.radians(self.orientation)
        new_x = self.pos[0] + distance * math.cos(rad)
        new_y = self.pos[1] + distance * math.sin(rad)
        new_pos = (new_x, new_y)
        
        if not self.detect_collision(new_pos):
            self.model.space.move_agent(self, new_pos)
            self.pos = new_pos

    def turn_left(self, angle):
        """Turn the robot left by a specified angle."""
        self.orientation = (self.orientation - angle) % 360

    def turn_right(self, angle):
        """Turn the robot right by a specified angle."""
        self.orientation = (self.orientation + angle) % 360

    def step(self):
        """Advance the robot by one step."""
        # Example behaviour: move forward and randomly turn
        self.move_forward(self.move_speed)  # Move forward by 1 unit
        if self.model.random.random() < 0.5:
            self.turn_left(self.turn_speed)  # Turn left by 15 degrees
        else:
            self.turn_right(self.turn_speed)  # Turn right by 15 degrees
        
        # Attempt to detect casualties within vision_range.
        self.detect_casualties()

    def detect_collision(self, new_pos):
        """Check collisions with walls and other robot agents."""
        # Check for wall collisions
        for agent in self.model.agents:
            if isinstance(agent, WallAgent):
                if self._collides_with_wall(new_pos, agent):
                    return True
                # elif type(agent).__name__ == "RobotAgent" and agent.unique_id != self.unique_id:
                #     if euclidean_distance(new_pos, agent.pos) < (self.radius * 2):
                #         return True

        # Check if the new position is out of bounds
        x, y = new_pos
        if x < 0 or x > self.model.width or y < 0 or y > self.model.height:
            return True

        return False

    def _collides_with_wall(self, new_pos, wall_agent):
        """
        Collision detection for a circular robot and a rectangular wall.
        wall_agent.wall_spec holds the wall rectangle information.
        """
        px, py = new_pos
        spec = wall_agent.wall_spec
        wx, wy = spec['x'], spec['y']
        half_w = spec['width'] / 2
        half_h = spec['height'] / 2

        # Find the closest point on the rectangle to the circle centre.
        closest_x = max(wx - half_w, min(px, wx + half_w))
        closest_y = max(wy - half_h, min(py, wy + half_h))

        # Calculate the distance from the agent's position to the closest point on the wall
        distance = euclidean_distance(new_pos, (closest_x, closest_y))
        return distance < self.radius

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