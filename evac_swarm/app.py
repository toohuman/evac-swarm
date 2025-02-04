from evac_swarm.visualization.ModularServer import ModularServer
from evac_swarm.visualization.modules import CanvasGrid
from model import BuildingModel

def portray_agent(agent):
    if agent.__class__.__name__ == "RobotAgent":
        return {"Shape": "circle", "r": 1, "Colour": "blue", "Layer": 2}
    elif agent.__class__.__name__ == "WallAgent":
        return {"Shape": "rect", "w": agent.wall_spec["width"], "h": agent.wall_spec["height"], 
                "Colour": "black", "Layer": 1}
    elif agent.__class__.__name__ == "CasualtyAgent":
        colour = "red" if not agent.discovered else "green"
        return {"Shape": "circle", "r": 1, "Colour": colour, "Layer": 2}
    return {}

# CanvasGrid for continuous space might require a custom visualiser; for now we use a simple grid.
grid = CanvasGrid(portray_agent, 100, 100, 500, 500)

server = ModularServer(
    BuildingModel,
    [grid],
    "Robotic Swarm Simulation",
    {"width": 100, "height": 100, "robot_count": 20, "casualty_count": 5,
     "min_room_size": 20, "wall_thickness": 1, "vision_range": 15}
)

if __name__ == "__main__":
    server.port = 8521  # Default port
    server.launch() 