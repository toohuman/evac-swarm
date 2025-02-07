from evac_swarm.agents import RobotAgent, WallAgent, CasualtyAgent
from evac_swarm.model import SwarmExplorerModel
from mesa.visualization import SolaraViz, Slider
from mesa.visualization.components import make_space_component, make_plot_component
import solara
import numpy as np
import matplotlib.patches as patches
from mesa.experimental.devs import ABMSimulator
from functools import partial

def agent_portrayal(agent):
    """Define how to portray each type of agent"""
    if agent is None:
        return {}
    
    if isinstance(agent, RobotAgent):
        return {
            "color": "blue",
            "marker": "o",  # Circle marker
            "size": 50     # Size in points
        }
        
    elif isinstance(agent, WallAgent):
        return {
            "color": "grey",
            "marker": "s",  # Square marker
            "size": agent.wall_spec["width"] * 100  # Scale up the size
        }
        
    elif isinstance(agent, CasualtyAgent):
        return {
            "color": "red" if not agent.discovered else "green",
            "marker": "o",
            "size": 30
        }

def post_process_with_sim(ax, sim):
    return post_process_space(ax, sim.model)

@solara.component
def post_process_space(ax, model):
    """Post-process the space visualization to add walls.
    Instead of clearing the entire axes (which resets the view), we remove existing wall patches.
    Then we re-draw the walls and re-apply the axis limits.
    """
    
    # We assume that the wall patches have facecolor 'grey'; adjust the filter as needed.
    for patch in list(ax.patches):
        if patch.get_facecolor()[:3] == (0.5, 0.5, 0.5) or patch.get_facecolor()[:3] == (0.7529411764705882, 0.7529411764705882, 0.7529411764705882):  
            # The exact grey value may depend on your parameters.
            patch.remove()

    # Draw walls as rectangles using the latest wall specifications:
    for wall_spec in model.space.wall_specs:
        x = wall_spec['x'] - wall_spec['width'] / 2
        y = wall_spec['y'] - wall_spec['height'] / 2
        rect = patches.Rectangle(
            (x, y),
            wall_spec['width'],
            wall_spec['height'],
            facecolor='grey',
            edgecolor='black',
            alpha=0.8
        )
        ax.add_patch(rect)

    # Reinstate the axis limits so the view stays correct.
    ax.set_xlim(0, model.width)
    ax.set_ylim(0, model.height)
    ax.set_xticks([])
    ax.set_yticks([])

# Model parameters with explicit value extraction
model_params = {
    "width": 40,  # Building is 1.5x larger than original dimensions (15 * 1.5)
    "height": 25,
    "robot_count": Slider("Robots", 5, 1, 20).value,
    "casualty_count": Slider("Casualties", 3, 1, 10).value,
    "min_room_size": 4,
    "wall_thickness": 0.5,
    "vision_range": Slider("Vision Range", 3, 1, 10).value
}

# Create a simulator that will re-instantiate the model on reset.
simulator = ABMSimulator()

# Instantiate the model via the simulator
model = SwarmExplorerModel(
    **model_params,
    simulator=simulator
)

# Create components with model reference
space = make_space_component(
        agent_portrayal,
        post_process=partial(post_process_with_sim, sim=simulator),
        space_name="space",
        canvas_width=600,
        canvas_height=600,
        grid_width=int(round(model.width)),
        grid_height=int(round(model.height)),
        draw_grid=False,
    )

coverage_chart = make_plot_component(
    "Coverage",
    backend="matplotlib"
)

# Create the Solara-based visualisation, passing the simulator so resets create a new model instance.
Page = SolaraViz(
    model=model,
    components=[space, coverage_chart],
    name="Evacuation Robot Swarm",
    model_params=model_params,
    simulator=simulator,
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