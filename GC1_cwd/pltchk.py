import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_training_progress():
    try:
        # Read data - skip header if present
        df = pd.read_csv('Training_Values/8_16_5_100.0_0.0001_1e-05_metrics_sumrate.csv',
                        header=None,
                        names=['episode', 'step', 'reward'],
                        skiprows=1)  # Skip header row
        
        # Convert reward column to float
        df['reward'] = pd.to_numeric(df['reward'], errors='coerce')
        
        # Drop any NaN values
        df = df.dropna()
        
        # Setup plot
        plt.style.use('seaborn')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Raw data plot
        ax1.plot(df['step'], df['reward'], 'b-', alpha=0.3, label='Raw Reward')
        ax1.set_title('Raw Training Progress')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Sum Rate (bps/Hz)')
        ax1.grid(True)
        ax1.legend()
        
        # Smoothed plot
        window = 10000
        rolling_mean = df['reward'].rolling(window=window).mean()
        rolling_std = df['reward'].rolling(window=window).std()
        
        ax2.plot(df['step'], rolling_mean, 'r-', linewidth=2, 
                label=f'Moving Average (window={window})')
        ax2.fill_between(df['step'],
                        rolling_mean - rolling_std,
                        rolling_mean + rolling_std,
                        alpha=0.2,
                        color='red',
                        label='±1 std dev')
        
        ax2.set_title('Smoothed Training Progress')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Sum Rate (bps/Hz)')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("First few lines of data:")
        print(pd.read_csv('Training_Values/8_16_5_100.0_0.0001_1e-05_metrics_sumrate.csv', nrows=5))

if __name__ == "__main__":
    plot_training_progress()