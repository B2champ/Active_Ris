import matplotlib.pyplot as plt
import pandas as pd
import time
import os

# Path to the CSV file
csv_file_path = "Average_Reward.csv"  # Change to the correct path of your CSV file

def plot_rewards(csv_path):
    """
    Continuously plot the reward values from a CSV file.
    """
    plt.ion()  # Turn on interactive mode for live updating
    fig, ax = plt.subplots()
    ax.set_title("Live Reward Plot")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Reward")

    line, = ax.plot([], [], label="Reward")
    ax.legend()

    episodes = []
    rewards = []

    last_updated = 0  # To track the last modification time of the file

    try:
        while True:
            # Check if the file exists and has been updated
            if os.path.exists(csv_path):
                new_updated = os.path.getmtime(csv_path)
                if new_updated != last_updated:
                    print("File has been updated.")
                    # Read the updated data
                    data = pd.read_csv(csv_path, header=None)
                    print(f"Data from CSV: {data.head()}")  # Print the first few rows of the data
                    episodes = range(1, len(data) + 1)
                    rewards = data[0].values  # Assuming reward values are in the first column

                    # Update the plot
                    line.set_xdata(episodes)
                    line.set_ydata(rewards)
                    ax.relim()
                    ax.autoscale_view()

                    plt.draw()
                    plt.pause(0.1)

                    last_updated = new_updated
            else:
                print(f"Waiting for file: {csv_path}")

            time.sleep(1)  # Wait before checking again

    except KeyboardInterrupt:
        print("Live plotting stopped.")
    finally:
        plt.ioff()
        plt.show()

# Run the plotting function
if __name__ == "__main__":
    plot_rewards(csv_file_path)
