import random

def generate_building_layout(width, height, min_room_size=20, wall_thickness=1):
    """
    Generate a building layout using Binary Space Partitioning.
    Returns a list of wall specifications (each a dict with x, y, width and height).
    """
    walls = []
    
    # Add outer walls
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
        # Check if the area is large enough to partition.
        if w < 2 * min_room_size or h < 2 * min_room_size:
            return
        
        # Randomly choose orientation based on dimensions.
        if w >= h:
            orientation = "vertical"
        else:
            orientation = "horizontal"
            
        if orientation == "vertical":
            # Choose an x coordinate for the split
            split_x = random.uniform(x + min_room_size, x + w - min_room_size)
            # Choose door position along the wall (avoid corners).
            door_y = random.uniform(y + h * 0.3, y + h * 0.7)
            
            # Add two wall segments with a gap for the door.
            # Bottom segment
            segment_height = door_y - y - (wall_thickness / 2)
            if segment_height > 0:
                walls.append({
                    "x": split_x,
                    "y": y + segment_height / 2,
                    "width": wall_thickness,
                    "height": segment_height
                })
            # Top segment
            segment_height = (y + h) - door_y - (wall_thickness / 2)
            if segment_height > 0:
                walls.append({
                    "x": split_x,
                    "y": door_y + segment_height / 2,
                    "width": wall_thickness,
                    "height": segment_height
                })
            
            # Recursively partition the left and right areas.
            partition(x, y, split_x - x, h)
            partition(split_x, y, x + w - split_x, h)
        
        else:
            # Horizontal split
            split_y = random.uniform(y + min_room_size, y + h - min_room_size)
            door_x = random.uniform(x + w * 0.3, x + w * 0.7)
            
            # Left segment
            segment_width = door_x - x - (wall_thickness / 2)
            if segment_width > 0:
                walls.append({
                    "x": x + segment_width / 2,
                    "y": split_y,
                    "width": segment_width,
                    "height": wall_thickness
                })
            # Right segment
            segment_width = (x + w) - door_x - (wall_thickness / 2)
            if segment_width > 0:
                walls.append({
                    "x": door_x + segment_width / 2,
                    "y": split_y,
                    "width": segment_width,
                    "height": wall_thickness
                })
            
            # Recursively partition the top and bottom areas.
            partition(x, y, w, split_y - y)
            partition(x, split_y, w, y + h - split_y)
    
    partition(0, 0, width, height)
    return walls 

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Set parameters for the building layout
    width = 100
    height = 100
    min_room_size = 20
    wall_thickness = 1

    # Generate the building layout
    layout = generate_building_layout(width, height, min_room_size, wall_thickness)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')

    # Draw each wall as a rectangle
    for wall in layout:
        # The wall is defined with its centre at (wall['x'], wall['y'])
        # and has dimensions wall['width'] and wall['height'].
        lower_left = (wall['x'] - wall['width'] / 2, wall['y'] - wall['height'] / 2)
        rect = patches.Rectangle(lower_left, wall['width'], wall['height'],
                                 edgecolor='black', facecolor='gray')
        ax.add_patch(rect)

    plt.title("Generated Building Layout")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show() 