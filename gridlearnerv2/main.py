# main.py
from matplotlib import pyplot as plt
from editor import RLMain
from config import Config

if __name__ == "__main__":
    # Create and display the grid editor, passing the grid size from the config
    print("Launching RLMain")
    m = RLMain()
    plt.show()
    print("Application closed.")