# main.py
from matplotlib import pyplot as plt
from editor import GridEditor
from config import Config

if __name__ == "__main__":
    # config instance
    config = Config()
    # Create and display the grid editor, passing the grid size from the config
    print("Launching Grid Editor")
    editor = GridEditor(grid_size=config.grid_size)
    plt.show()
    print("Application closed.")