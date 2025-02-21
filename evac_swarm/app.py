from evac_swarm.agents import RobotAgent, WallAgent, CasualtyAgent, DeploymentAgent
from evac_swarm.model import SwarmExplorerModel
from mesa.visualization import SolaraViz, Slider
from mesa.visualization.components import make_space_component, make_plot_component
import solara
import numpy as np
import matplotlib.patches as patches
from mesa.experimental.devs import ABMSimulator
from functools import partial
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm

# Global mapping of species to their colours
COLOR_MAPPING = {
    "Robot": "#9467bd",
    "Casualty": {"default": "#d62728", "discovered": "#2ca02c"},
    "Wall": "grey",
    "Deployment": "#1f77b4"  # Add a colour for the DeploymentAgent
}

def agent_portrayal(agent):
    """Define how to portray each type of agent"""
    if agent is None:
        return {}
    
    if isinstance(agent, RobotAgent):
        return {
            "color": COLOR_MAPPING["Robot"],
            "marker": "o",  # Circle marker
            "size": agent.radius * 100,  # Scale the radius for clear visualisation
        }
        
    elif isinstance(agent, WallAgent):
        return {
            "color": COLOR_MAPPING["Wall"],
            "marker": "s",  # Square marker
            "size": agent.wall_spec["width"] * 100  # Scale up the size
        }
        
    elif isinstance(agent, CasualtyAgent):
        casualty_color = COLOR_MAPPING["Casualty"]["discovered"] if agent.discovered else COLOR_MAPPING["Casualty"]["default"]
        return {
            "color": casualty_color,
            "marker": "o",
            "size": 50
        }

    elif isinstance(agent, DeploymentAgent):
        return {
            "color": COLOR_MAPPING["Deployment"],
            "marker": "D",  # Diamond marker
            "size": 100  # Fixed size for visibility
        }

def post_process_with_sim(ax, sim):
    return post_process_space(ax, sim.model)

@solara.component
def post_process_space(ax, model):
    """Post-process the space visualization to add walls.
    Instead of clearing the entire axes (which resets the view), we remove existing wall patches.
    Then we re-draw the walls and re-apply the axis limits.
    """
    # Display the coverage grid as a semi-transparent overlay, behind other layers.
    if hasattr(model, "coverage_grid"):
        # Create a custom colormap that maps:
        #   -1 (wall) -> black, 0 (not visited) -> black, 1 (visited) -> white.
        cmap = ListedColormap(['black', '#111111', 'white'])
        norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
        ax.imshow(
            model.coverage_grid,
            extent=[0, model.width, 0, model.height],
            origin="lower",
            cmap=cmap,
            norm=norm,
            alpha=1.0,
            zorder=0
        )
    
    # Draw walls as rectangles using the latest wall specifications:
    for wall_spec in model.space.wall_specs:
        x = wall_spec['x'] - wall_spec['width'] / 2
        y = wall_spec['y'] - wall_spec['height'] / 2
        rect = patches.Rectangle(
            (x, y),
            wall_spec['width'],
            wall_spec['height'],
            facecolor='black',
            edgecolor='none',
            alpha=0.8
        )
        ax.add_patch(rect)
    
    # Draw vision/communication range for each RobotAgent.
    if model.show_vision_range:
        for agent in model.agents:
            if isinstance(agent, RobotAgent):
                x, y = agent.pos
                # Create a circle patch with radius equal to the agent's vision range.
                vision_circle = patches.Circle(
                    (x, y),
                    agent.vision_range,
                    fill=False,
                    edgecolor='black',
                    linestyle='--',
                    linewidth=1,
                    alpha=0.5
                )
                ax.add_patch(vision_circle)

    # Reinstate the axis limits and enforce an equal aspect ratio so circles appear round.
    ax.set_xlim(0, model.width)
    ax.set_ylim(0, model.height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])

@solara.component
def post_process_coverage(ax):
    """Post-process the coverage chart to set the y-axis fixed between 0 and 100."""
    ax.set_ylim(0, 100)

# Model parameters with explicit value extraction
model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "use_seed": {
        "type": "Checkbox",
        "value": False,
        "label": "Use Fixed Seed",
    },
    "move_behaviour": {
        "type": "Select",
        "value": "random",
        "values": ["random", "disperse"],
        "label": "Movement Behaviour"
    },
    "show_vision_range": {
        "type": "Checkbox",
        "value": True,
        "label": "Show Vision Range",
    },
    "width": {
        'type': "InputText",
        'value': 40,
        'label': "Building Width"
    },
    "height": {
        'type': "InputText",
        'value': 25,
        'label': "Building Height"
    },
    "min_room_size": 
    {
        'type': "InputText",
        'value': 6,
        'label': "Minimum Room Size"
    },
    "wall_thickness": Slider("Wall Thickness", 0.5, 0.1, 0.8, step=0.01),
    "robot_count": Slider("Robots", 20, 1, 50),
    "casualty_count": Slider("Casualties", 3, 1, 10),
    "vision_range": Slider("Vision Range", 2, 1, 10),
}

# Create a simulator that will re-instantiate the model on reset.
simulator = ABMSimulator()

# Instantiate the model via the simulator
model = SwarmExplorerModel(
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
    backend="matplotlib",
    post_process=post_process_coverage
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