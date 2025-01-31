import jax
import jax.numpy as jnp

class Environment:
    def __init__(self, width=100, height=100):
        self.width = width
        self.height = height
        self.walls = jnp.array([], dtype=jnp.float32)  # [[x1,y1,x2,y2], ...]
        self.robots = []
        self.casualties = jnp.array([], dtype=jnp.float32)  # [ [x,y], ... ]
        
    def add_wall(self, x1, y1, x2, y2):
        self.walls = jnp.vstack([self.walls, jnp.array([x1,y1,x2,y2])]) if self.walls.size else jnp.array([[x1,y1,x2,y2]])
        
    def check_collision(self, position, radius=0.5):
        """Check collision between circular agent and walls"""
        # Vectorized wall collision check using JAX
        def line_segment_distance(wall):
            # Wall: [x1,y1,x2,y2], Position: [x,y]
            p = position[:2]
            a = wall[:2]
            b = wall[2:]
            pa = p - a
            ba = b - a
            t = jnp.clip(jnp.dot(pa, ba) / jnp.dot(ba, ba), 0, 1)
            closest = a + t * ba
            return jnp.linalg.norm(p - closest)
        
        distances = jax.vmap(line_segment_distance)(self.walls)
        return jnp.any(distances < radius) 