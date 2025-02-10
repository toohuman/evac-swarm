import random
import matplotlib.patches as patches

def generate_building_layout(
    width, height,
    min_room_size=3, wall_thickness=0.3,
    rng=None
):
    """
    Generate a building layout using Binary Space Partitioning.
    Returns a list of wall specifications (each a dict with x, y, width and height).
    """
    # Set random seed if provided
    if rng is None:
        print("UH OH")
        rng = random.Random()

    # Define real-world scale for internal features
    scale_factor = 1.0  # 1 unit = 1 meter

    # Convert internal dimensions to real-world scale
    min_room_size *= scale_factor
    wall_thickness *= scale_factor
    door_width = 1 * scale_factor

    walls = []
    
    # Add outer walls (fixed dimensions)
    walls.append({
        "x": width / 2,
        "y": wall_thickness / 2,
        "width": width,
        "height": wall_thickness
    })
    walls.append({
        "x": width / 2,
        "y": height - wall_thickness / 2,
        "width": width,
        "height": wall_thickness
    })
    walls.append({
        "x": wall_thickness / 2,
        "y": height / 2,
        "width": wall_thickness,
        "height": height
    })
    walls.append({
        "x": width - wall_thickness / 2,
        "y": height / 2,
        "width": wall_thickness,
        "height": height
    })
    
    def partition(x, y, w, h):
        """
        Recursively partition the area and add inner walls with door gaps.
        """
        if w < 2 * min_room_size or h < 2 * min_room_size:
            return
        
        if w >= h:
            orientation = "vertical"
        else:
            orientation = "horizontal"
            
        if orientation == "vertical":
            split_x = rng.uniform(x + min_room_size, x + w - min_room_size)
            door_y = rng.uniform(y + h * 0.4, y + h * 0.6)  # Centralize door
            
            # Ensure segments align at the door
            segment_height = door_y - y - door_width / 2
            if segment_height > 0:
                walls.append({
                    "x": split_x,
                    "y": y + segment_height / 2,
                    "width": wall_thickness,
                    "height": segment_height  # Extend to meet other walls
                })
            segment_height = (y + h) - door_y - door_width / 2
            if segment_height > 0:
                walls.append({
                    "x": split_x,
                    "y": door_y + door_width / 2 + segment_height / 2,
                    "width": wall_thickness,
                    "height": segment_height  # Extend to meet other walls
                })
            
            partition(x, y, split_x - x, h)
            partition(split_x, y, x + w - split_x, h)
        
        else:
            split_y = rng.uniform(y + min_room_size, y + h - min_room_size)
            door_x = rng.uniform(x + w * 0.4, x + w * 0.6)  # Centralize door
            
            # Ensure segments align at the door
            segment_width = door_x - x - door_width / 2
            if segment_width > 0:
                walls.append({
                    "x": x + segment_width / 2,
                    "y": split_y,
                    "width": segment_width + wall_thickness / 2,  # Extend to meet other walls
                    "height": wall_thickness
                })
            segment_width = (x + w) - door_x - door_width / 2
            if segment_width > 0:
                walls.append({
                    "x": door_x + door_width / 2 + segment_width / 2,
                    "y": split_y,
                    "width": segment_width + wall_thickness / 2,  # Extend to meet other walls
                    "height": wall_thickness
                })
            
            partition(x, y, w, split_y - y)
            partition(x, split_y, w, y + h - split_y)
    
    partition(0, 0, width, height)
    return walls

def draw_walls(ax, walls):
    for wall in walls:
        lower_left = (wall['x'] - wall['width'] / 2, wall['y'] - wall['height'] / 2)
        rect = patches.Rectangle(lower_left, wall['width'], wall['height'],
                                 edgecolor='black', facecolor='gray')
        ax.add_patch(rect)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Set parameters for the building layout
    width = 15
    height = 15
    min_room_size = 3
    wall_thickness = 0.3

    # Generate the building layout
    layout = generate_building_layout(width, height, min_room_size, wall_thickness)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')

    # Draw each wall as a rectangle
    draw_walls(ax, layout)

    plt.title("Generated Building Layout")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show() 