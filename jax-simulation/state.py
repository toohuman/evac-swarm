from dataclasses import dataclass
import jax
import jax.numpy as jnp

@dataclass
class RobotState:
    positions: jax.Array  # [N, 2] array
    directions: jax.Array  # [N, 2] array (unit vectors)
    speeds: jax.Array  # [N] array
    comm_ranges: jax.Array  # [N] array
    in_network: jax.Array  # [N] bool array (connected to base)
    
@dataclass
class SimState:
    robots: RobotState
    walls: jax.Array  # [M, 4] array of line segments
    casualties: jax.Array  # [K, 2] array
    exploration: jax.Array  # 2D grid of explored areas
    time: float 