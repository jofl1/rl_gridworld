# Reinforcement Learning Gridworld

This project explores the implementation of a Q-learning agent in a grid-based environment. The agent's goal is to navigate from a starting position to a goal, avoiding penalties along the way (fires). The project demonstrates several iterations of the implementation, from a simple script to a more complex, interactive application.

## Project Structure

The project is organised into several files, each representing a different stage of development or a specific component of the final application.

### Legacy Implementations

These files represent the initial development stages of the project and are kept for reference.

*   `gridlearner.py`: The first implementation of the Q-learning algorithm. This is a simple, self-contained script that demonstrates the core logic.
*   `gridlearner_classesv0.py`: The original implementation refactored into a class-based structure. This version introduces classes for `Config`, `Action`, `State`, `World`, `Agent`, `Trainer`, and `Visualiser`.
*   `gridlearner_classesv1.py`: An evolution of the class-based structure, this version incorporates a more advanced visualiser using `matplotlib.animation` to show the agent's learning process in real-time.

### Current Implementation (`gridlearnerv2`)

This directory contains the most recent and feature-rich version of the project. It is a more modular and interactive application that allows the user to design their own grid-world environments.

*   `main.py`: The entry point for the application. It launches the grid editor.
*   `editor.py`: An interactive grid editor built with `matplotlib`. It allows the user to design the grid-world by placing walls, the start and goal positions, and fire (penalty) cells.
*   `config.py`: Contains the configuration for the application, including the default grid-world layout and Q-learning hyperparameters.
*   `simulation_core.py`: The core logic for the Q-learning simulation. It includes the `Action`, `State`, `World`, `Agent`, and `Trainer` classes.
*   `visualisers.py`: Contains the `Visualiser` and `LiveVisualiser` classes, which are responsible for displaying the final results and the live training process, respectively.

## How to Run

To run the latest version of the application, execute the `main.py` script in the `gridlearnerv2` directory:

```bash
python gridlearnerv2/main.py
```

Currently editor is 'redundant' due to using preconfig of walls (see blockworld1 in config.py.