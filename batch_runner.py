"""
Batch runner experiments for the SwarmExplorerModel.

This script sweeps over:
  - The number of robots: 5, 10, 15, and 20
  - The movement behaviour: "disperse" vs "random"

Each experiment runs until either a fixed number of steps (max_steps)
is reached or until the coverage (computed as the percentage of accessible grid cells
that have been visited) reaches 100%. The runtime for each experiment is recorded.

Usage: Run as a normal python file. For example:
    python batch_run_experiments.py
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from evac_swarm.model import SwarmExplorerModel
from tqdm.auto import tqdm

# Define the parameter sweep.
params = {
    "width": 80,
    "height": 45,
    "robot_count": [10, 20, 30, 40, 50],
    "casualty_count": 5,
    "min_room_size": 6,
    "wall_thickness": 0.5,
    "vision_range": 3,
    "move_behaviour": ["disperse", "random"],
    # Add additional parameters here if needed.
}

def run_experiment(run_id, iteration, kwargs, max_steps=10000):
    """
    Run a single model instance until either max_steps
    are reached or the coverage reaches at least 100%.
    The function records the number of steps taken and the runtime.

    Returns:
        dict: A summary of the run including the run id, parameters,
              final step count, final coverage, and run time.
    """
    # Instantiate the model with the given parameters.
    model = SwarmExplorerModel(**kwargs)
    start_time = time.time()
    # Create a tqdm progress bar for the steps in this run.
    pbar = tqdm(total=max_steps, desc=f"Run {run_id} Iteration {iteration}", leave=False)
    # Run until either the maximum number of steps is reached or coverage is complete.
    while model.running and model.steps <= max_steps:
        model.step()
        # Update the progress bar for each step.
        pbar.update(1)
        # coverage = (np.sum(model.coverage_grid) / model.total_accessible_cells) * 100
        deployment_coverage = model.get_deployment_coverage_percentage()
        if deployment_coverage >= 100:
            break
    pbar.close()
    end_time = time.time()
    run_time = end_time - start_time

    # Create a result summary.
    result = {
        "RunId": run_id,
        "Iteration": iteration,
        **kwargs,
        "Steps": model.steps,
        "FinalCoverage": coverage,
        "RunTime": run_time,
    }
    return result

def make_runs(parameters, iterations=1):
    """
    Generate a list of runs given the parameter sweep.

    Each run is represented as a tuple: (run_id, iteration, kwargs)
    """
    runs = []
    run_id = 0
    keys = list(parameters.keys())
    # Build the Cartesian product over parameters.
    values_product = product(
        *(parameters[k] if isinstance(parameters[k], list) else [parameters[k]] for k in keys)
    )
    for iteration in range(iterations):
        for vals in values_product:
            kwargs = dict(zip(keys, vals))
            runs.append((run_id, iteration, kwargs))
            run_id += 1
    return runs

if __name__ == "__main__":
    iterations = 1  # Number of repetitions per parameter combination.
    max_steps = 1000
    runs = make_runs(params, iterations=iterations)
    results = []
    # Wrap the loop with tqdm to show a progress bar.
    for run in tqdm(runs, total=len(runs), desc="Running experiments"):
        result = run_experiment(*run, max_steps=max_steps)
        results.append(result.copy())

    # Create a pandas dataframe from the results list of dictionaries.
    df = pd.DataFrame(results)

    # Plot coverage % against number of robots, with lines for each exploration strategy.
    fig, ax = plt.subplots()
    for strategy in df['move_behaviour'].unique():
        # Use the correct column for the number of robots: "robot_count"
        strategy_df = df[df['move_behaviour'] == strategy].sort_values('robot_count')
        ax.plot(strategy_df['robot_count'], strategy_df['FinalCoverage'],
                label=strategy, marker='o')

    ax.set_xlabel('Number of Robots')
    ax.set_ylabel('Coverage (%)')
    ax.set_title('Coverage % vs Number of Robots')
    ax.legend(title='Exploration Strategy')
    plt.show()