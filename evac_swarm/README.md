# Robotic Swarm Simulation

This project is a proof-of-concept simulation of a robotic swarm deployed in a post-natural disaster recovery scenario. The simulation uses the MESA framework in a continuous 2D space.

## Description

- **Agents:**
  - **RobotAgent:** A small, circular robot with directional movement and collision detection. All robots follow a shared policy which is trained online using reinforcement learning.
  - **CasualtyAgent:** Stationary agents representing casualties which the robots must detect and report.
  - **WallAgent:** Impassable obstacles representing walls.

- **Building Generation:**  
  The building floor plan is generated using a basic Binary Space Partitioning algorithm. Rooms are of a specified minimum size and include doorways (not placed in corners) to ensure connectivity throughout the building.
  
- **Communication Constraints:**  
  Each robot has a limited vision/communication range with line-of-sight constraints. They must maintain network connectivity with the deployment operator at the point-of-entry (by default, the building entry).

## Requirements

- Python 3.8+
- Mesa
- Other dependencies as needed (e.g. for implementing RL with torchRL)

## Usage

- To **run a single simulation**, execute:
  ```
  python run.py
  ```

- To **launch the visualisation server**, execute:
  ```
  python app.py
  ```

## Project Structure

- `README.md`                : This file.
- `model.py`                 : The main MESA model.
- `agents.py`                : Definitions for Robot, Wall, and Casualty agents.
- `building_generator.py`    : Building (floor plan) generation using BSP.
- `run.py`                   : Script to run the simulation.
- `app.py`                   : (Optional) Visualisation server code for Solara-based GUI. 