from mesa.visualization import SolaraViz, Slider
from mesa.visualization.components import make_space_component, make_plot_component
import solara

from evac_swarm.model import SwarmExplorerModel

def agent_portrayal(agent):
    """Define how to portray each type of agent"""
    portrayal = {}
    
    if agent.__class__.__name__ == "RobotAgent":
        portrayal = {
            "shape": "circle",
            "color": "blue",
            "filled": True,
            "layer": 1,
            "radius": 0.5,
        }
        
    elif agent.__class__.__name__ == "WallAgent":
        portrayal = {
            "shape": "rect",
            "color": "grey",
            "filled": True,
            "layer": 0,
            "width": agent.wall_spec["width"],
            "height": agent.wall_spec["height"]
        }
        
    elif agent.__class__.__name__ == "CasualtyAgent":
        portrayal = {
            "shape": "circle",
            "color": "red" if not agent.discovered else "green",
            "filled": True,
            "layer": 1,
            "radius": 0.3,
        }
        
    return portrayal

# Model parameters with explicit value extraction
model_params = {
    "width": 100,
    "height": 100,
    "robot_count": Slider("Robots", 20, 1, 50).value,
    "casualty_count": Slider("Casualties", 5, 1, 20).value,
    "min_room_size": 20,
    "wall_thickness": 1,
    "vision_range": Slider("Vision Range", 15, 5, 30).value
}

# Create space component with explicit model reference
space = make_space_component(
    agent_portrayal,
    space_name="space",
    canvas_width=500,
    canvas_height=500,
    grid_width=100,
    grid_height=100
)

# Create chart component with matplotlib backend
coverage_chart = make_plot_component(
    "Coverage",
    backend="matplotlib"
)

# Model
model = SwarmExplorerModel()

# Create the Solara-based visualisation with explicit instantiation
Page = SolaraViz(
    model=model,
    components=[space, coverage_chart],
    name="Evacuation Robot Swarm",
    model_params=model_params
)

# --- Launching the app ---
if __name__ == "__main__":
    # If running from Python directly, try:
    try:
        solara.run_app(Page)
    except AttributeError:
        print("solara.run_app is not available in your Solara installation.")
        print("Please run the app by setting the environment variable and using:")
        print("    export SOLARA_APP=evac_swarm.app")
        print("    solara run evac_swarm.app") 