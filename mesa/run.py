from model import BuildingModel

if __name__ == "__main__":
    # Create the model with default parameters.
    model = BuildingModel(width=100, height=100, robot_count=20, casualty_count=5,
                          min_room_size=20, wall_thickness=1, vision_range=15)
    
    for i in range(200):
        model.step()
        # Optional: insert logging or print statements to track progress.
        if i % 20 == 0:
            print(f"Step {i}: Simulation running...") 