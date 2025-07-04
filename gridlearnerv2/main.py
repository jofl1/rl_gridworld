# main.py
from matplotlib import pyplot as plt
from editor import GridEditor

if __name__ == "__main__":
    # Create and display the grid editor. The editor will handle the rest.
    print("Launching Grid Editor")
    editor = GridEditor(grid_size=10)
    plt.show()
    print("Application closed.")