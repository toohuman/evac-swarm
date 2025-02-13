# Robot Swarm Explorer

A simulation of autonomous robot swarms exploring buildings to locate casualties following natural disasters. Built using Mesa (an agent-based modelling framework) and Solara for interactive visualisation.

## Overview

This project simulates multiple robots coordinating to explore an unknown building layout and locate casualties. It demonstrates:

- Autonomous robot swarm behaviour
- Real-time coverage mapping with line-of-sight constraints
- Procedurally generated building layouts
- Interactive visualisation of the exploration process

### Key Features

- **Robot Agents:** Small, mobile robots with limited vision range and collision avoidance
- **Building Generation:** Procedural floor plan generation using Binary Space Partitioning
- **Coverage Tracking:** Real-time visualization of explored areas
- **Casualty Detection:** Simulated victims that robots must locate
- **Interactive Parameters:** Adjust robot count, vision range, building size, and more

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/robot-swarm-explorer.git
cd robot-swarm-explorer
```

2. Create and activate a virtual environment (recommended):

Using conda
```bash 
conda create -n robot-explorer python=3.9
conda activate robot-explorer
```

Or using venv
```bash
python -m venv venv
source venv/bin/activate # On Unix/macOS
```

3. Install dependencies:

```bash
pip install .
```

4. Run the web interface:

```bash
export PYTHONPATH=$PYTHONPATH:.
solara run evac_swarm.app
```

This will launch the simulation interface at http://localhost:8765

### Project Structure

evac_swarm/
├── app.py # Solara web application
├── model.py # Core simulation model
├── agents.py # Robot, Wall, and Casualty agents
├── space.py # Hybrid continuous/discrete space
└── building_generator.py # Procedural building generation

## Requirements

- Python 3.9 or higher
- Mesa 2.1.1 or higher (agent-based modelling)
- Solara 1.21.0 or higher (web interface)
- NumPy, SciPy (numerical computation)
- Matplotlib (visualization)
- Rtree (spatial indexing)

See `setup.py` for complete dependencies.

## Development

For development work, install additional tools: