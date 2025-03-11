from .model import BuildingModel

if __name__ == "__main__":
    # Create the model with default parameters.
    model = BuildingModel(width=80, height=42, robot_count=30, casualty_count=5,
                          min_room_size=6, wall_thickness=0.5, vision_range=3)
    
    for i in range(1000):
        model.step()
        # Optional: insert logging or print statements to track progress.
        if i % 20 == 0:
            print(f"Step {i}: Simulation running...") 