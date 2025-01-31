import jax
import jax.numpy as jnp
from .environment import Environment
from .robot import Robot

class Simulation:
    def __init__(self, env_config=None):
        self.environment = Environment()
        self.time = 0.0
        self.dt = 0.1  # Time step
        
    def add_robot(self, robot):
        self.environment.robots.append(robot)
        
    def step(self):
        """Advance simulation by one time step"""
        # Update robot positions
        for robot in self.environment.robots:
            proposed_pos = robot.move(robot.direction, self.dt)
            
            # Check collision before updating position
            if not self.environment.check_collision(proposed_pos):
                robot.position = proposed_pos
                
        # Update communication network
        self.update_communications()
        
        self.time += self.dt
        
    def update_communications(self):
        """Update communication graph between robots"""
        positions = jnp.array([r.position for r in self.environment.robots])
        for robot in self.environment.robots:
            comm_edges = robot.get_communication_edges(positions)
            # Store/process communication network as needed 