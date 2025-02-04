import jax
import jax.numpy as jnp

class Robot:
    def __init__(self, robot_id, position, direction, max_speed=1.0, comm_range=5.0):
        self.id = robot_id
        self.position = jnp.array(position, dtype=jnp.float32)  # [x, y]
        self.direction = jnp.array(direction, dtype=jnp.float32)  # unit vector
        self.max_speed = max_speed
        self.comm_range = comm_range
        self.current_speed = 0.0
        
    def move(self, new_direction, dt):
        """Update position based on movement direction and delta time"""
        # Normalise direction vector
        norm_dir = new_direction / jnp.linalg.norm(new_direction + 1e-6)
        self.direction = norm_dir
        self.current_speed = self.max_speed  # Simplified for now
        new_position = self.position + self.direction * self.current_speed * dt
        return new_position  # Position update to be validated by environment

    def get_communication_edges(self, other_positions):
        """Calculate which other robots are within communication range"""
        distances = jnp.linalg.norm(other_positions - self.position, axis=1)
        return distances < self.comm_range 