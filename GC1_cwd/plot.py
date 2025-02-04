import pandas as pd
import matplotlib.pyplot as plt

# List of CSV file names
file_names = ['max_ampli_action.csv','Max_Amplification.csv', 'Average_Reward.csv', 'Avg_PobingPower.csv']  # Add your file names here

for file_name in file_names:
    try:
        # Load the CSV file
        data = pd.read_csv(file_name)

        # Extract the column data (assuming first column is of interest)
        # column_name = data.columns[0] 
        # print(f'the first column name is {column_name}') # Get the first column name
        # values = data[column_name]  # Extract the column
        values = data  # Extract the column

        # Plot the graph
        plt.figure(figsize=(10, 6))
        plt.plot(values, marker='o', linestyle='--', color='b')
        # plt.plot(values, linestyle='dashdot', color='b')
        plt.title(f"Plot of {file_name}")
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.grid(True)

        # Save the plot
        output_file = file_name.replace('.csv', '.png')  # Save with same base name as the CSV
        plt.savefig( output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Plot saved as '{output_file}'")

    except Exception as e:
        print(f"Error processing file {file_name}: {e}")
