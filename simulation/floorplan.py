import jax
import jax.numpy as jnp
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

class FloorplanGenerator:
    def __init__(self, rng_key=jax.random.PRNGKey(0)):
        self.key = rng_key
        
    def generate(self, 
               perimeter: Tuple[float, float] = (100, 100),
               num_rooms: int = 5,
               min_room_size: float = 10.0) -> jnp.Array:
        """
        Generate connected rooms within given perimeter dimensions
        Returns array of walls in format [x1, y1, x2, y2]
        """
        self.key, subkey = jax.random.split(self.key)
        walls = []
        
        # Generate outer walls
        outer_walls = [
            [0, 0, perimeter[0], 0],          # Bottom wall
            [perimeter[0], 0, perimeter[0], perimeter[1]],  # Right wall
            [0, perimeter[1], perimeter[0], perimeter[1]],  # Top wall
            [0, 0, 0, perimeter[1]]           # Left wall
        ]
        
        # Generate internal rooms using recursive division
        rooms = self._recursive_division(
            jax.random.split(subkey)[0],
            bounds=(0, 0, perimeter[0], perimeter[1]),
            depth=0,
            max_rooms=num_rooms,
            min_size=min_room_size
        )
        
        # Collect all walls except doorways
        all_walls = outer_walls + [wall for room in rooms for wall in room.walls]
        return jnp.array(all_walls, dtype=jnp.float32)

    class Room:
        def __init__(self, bounds, doorways=None):
            self.bounds = bounds  # (x1, y1, x2, y2)
            self.walls = []
            self.doorways = doorways or []
            
            # Create walls but skip doorways
            x1, y1, x2, y2 = bounds
            full_walls = [
                (x1, y1, x2, y1),  # Bottom
                (x2, y1, x2, y2),  # Right
                (x1, y2, x2, y2),  # Top
                (x1, y1, x1, y2)   # Left
            ]
            
            for i, wall in enumerate(full_walls):
                # Check if this wall has a doorway
                if not any(dw['wall_idx'] == i for dw in self.doorways):
                    self.walls.append(wall)

    def _recursive_division(self, key, bounds, depth, max_rooms, min_size):
        current_key, split_key = jax.random.split(key)
        x1, y1, x2, y2 = bounds
        width = x2 - x1
        height = y2 - y1

        # Base case: room is too small or reached max rooms
        if (width < min_size*2 and height < min_size*2) or depth >= max_rooms:
            return [self.Room(bounds)]
            
        # Randomly choose split direction (0=horizontal, 1=vertical)
        split_dir = jax.random.randint(split_key, (), 0, 2)
        
        if split_dir == 0 and width >= 2*min_size:  # Vertical split
            split_pos = x1 + min_size + jax.random.uniform(current_key) * (width - 2*min_size)
            doorway_pos = y1 + jax.random.uniform(current_key) * (height - min_size/2)
            
            left_rooms = self._recursive_division(
                jax.random.split(key)[0],
                (x1, y1, split_pos, y2),
                depth+1,
                max_rooms,
                min_size
            )
            
            right_rooms = self._recursive_division(
                jax.random.split(key)[1],
                (split_pos, y1, x2, y2),
                depth+1,
                max_rooms,
                min_size
            )
            
            # Add doorway between the two sections
            doorway = {
                'wall_idx': 1,  # Right wall of left section
                'position': (split_pos, doorway_pos),
                'size': min_size/2
            }
            left_rooms[-1].doorways.append(doorway)
            
            return left_rooms + right_rooms
            
        elif split_dir == 1 and height >= 2*min_size:  # Horizontal split
            split_pos = y1 + min_size + jax.random.uniform(current_key) * (height - 2*min_size)
            doorway_pos = x1 + jax.random.uniform(current_key) * (width - min_size/2)
            
            bottom_rooms = self._recursive_division(
                jax.random.split(key)[0],
                (x1, y1, x2, split_pos),
                depth+1,
                max_rooms,
                min_size
            )
            
            top_rooms = self._recursive_division(
                jax.random.split(key)[1],
                (x1, split_pos, x2, y2),
                depth+1,
                max_rooms,
                min_size
            )
            
            # Add doorway between the two sections
            doorway = {
                'wall_idx': 2,  # Top wall of bottom section
                'position': (doorway_pos, split_pos),
                'size': min_size/2
            }
            bottom_rooms[-1].doorways.append(doorway)
            
            return bottom_rooms + top_rooms
        
        else:  # Can't split further
            return [self.Room(bounds)]

    @staticmethod
    def plot_walls(walls, ax=None):
        """Visualise the floorplan walls"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Convert walls to line segments
        segments = []
        for wall in walls:
            x1, y1, x2, y2 = wall
            segments.append([(x1, y1), (x2, y2)])
        
        lc = LineCollection(segments, color='black', linewidths=1)
        ax.add_collection(lc)
        
        ax.autoscale()
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Floorplan Layout')
        return ax

    @staticmethod
    def visualise(walls, ax=None):
        """Visualise the floorplan walls"""
        return FloorplanGenerator.plot_walls(walls, ax)

def image_to_floorplan(image_path):
    # Placeholder for CNN-based segmentation
    return {
        'walls': [[0,0,100,0], ...],
        'doorways': [...],
        'rooms': [...]
    }

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create generator and generate sample floorplan
    generator = FloorplanGenerator()
    walls = generator.generate(
        perimeter=(100, 80),
        num_rooms=7,
        min_room_size=8
    )
    
    # Plot and display
    generator.visualise(walls)
    plt.show()

