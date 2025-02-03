from functools import partial
import jax
import jax.numpy as jnp
from .state import SimState, RobotState

class JaxSimulator:
    def __init__(self, num_robots=10):
        self.key = jax.random.PRNGKey(0)
        self.state = self.initialize_state(num_robots)
        self.dt = 0.1
        
    def initialize_state(self, num_robots):
        return SimState(
            robots=RobotState(
                positions=jnp.zeros((num_robots, 2)),
                directions=jax.random.uniform(jax.random.PRNGKey(0), (num_robots, 2)),
                speeds=jnp.full(num_robots, 1.0),
                comm_ranges=jnp.full(num_robots, 5.0),
                in_network=jnp.zeros(num_robots, dtype=bool)
            ),
            walls=jnp.array([[0, 0, 100, 0], [100, 0, 100, 100]]),
            casualties=jnp.array([[80, 80], [20, 30]]),
            exploration=jnp.zeros((100, 100), dtype=bool),
            time=0.0
        )

    @partial(jax.jit, static_argnums=(0,))
    def update_step(self, state):
        """JIT-compiled update step using pure functions"""
        new_robots = self.update_robots(state)
        new_comm = self.update_communications(new_robots)
        return state.replace(
            robots=new_comm,
            time=state.time + self.dt
        )

    def update_robots(self, state):
        """Vectorized robot position update with collision checking"""
        # Proposed new positions for all robots
        new_positions = state.robots.positions + \
                      state.robots.directions * \
                      state.robots.speeds[:, None] * self.dt
        
        # Batch collision check for all robots
        collision_mask = jax.vmap(self.check_collision, in_axes=(0, None))(new_positions, state.walls)
        
        # Only update positions for robots without collisions
        safe_positions = jnp.where(collision_mask[:, None], state.robots.positions, new_positions)
        return state.robots.replace(positions=safe_positions)

    @staticmethod
    def check_collision(position, walls, radius=0.5):
        """Vectorized collision check between position and all walls"""
        def wall_distance(wall):
            a, b = wall[:2], wall[2:]
            pa = position - a
            ba = b - a
            t = jnp.clip(jnp.dot(pa, ba) / jnp.dot(ba, ba), 0, 1)
            closest = a + t * ba
            return jnp.linalg.norm(position - closest)
        
        distances = jax.vmap(wall_distance)(walls)
        return jnp.any(distances < radius)

    def update_communications(self, robots):
        """Update communication network using pairwise distances"""
        # All-to-all distance matrix
        diffs = robots.positions[:, None] - robots.positions
        distances = jnp.linalg.norm(diffs, axis=-1)
        
        # Communication adjacency matrix
        comm_matrix = (distances < robots.comm_ranges[:, None]) & (distances > 0)
        
        # Simple network connectivity check (needs proper implementation)
        return robots.replace(in_network=comm_matrix.any(axis=1))