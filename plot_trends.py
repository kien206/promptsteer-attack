import json
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    base_path = '/media/volume/DoubleSteeringLLM/promptsteer/LayerNavigator/test_result'
    positive_file = os.path.join(base_path, 'positive_test_results.json')
    baseline_file = os.path.join(base_path, 'baseline_test_results.json')
    negative_file = os.path.join(base_path, 'negative_test_results.json')
    attacked_file = os.path.join(base_path, 'attacked_test_results.json')
    attacked_dwb_file = os.path.join(base_path, 'attacked_dwb_test_results.json')
    output_dir = os.path.join(base_path, 'plots_v2')

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load data
    with open(positive_file, 'r') as f:
        positive_data = json.load(f)

    with open(baseline_file, 'r') as f:
        baseline_data = json.load(f)
    
    with open(negative_file, 'r') as f:
        negative_data = json.load(f)

    with open(attacked_file, 'r') as f:
        attacked_data = json.load(f)
    
    with open(attacked_dwb_file, 'r') as f:
        attacked_dwb_data = json.load(f)
    # Get all unique tasks from both files
    tasks = set(positive_data.keys()) | set(baseline_data.keys()) | set(negative_data.keys()) | set(attacked_data.keys())

    for task in tasks:
        plt.figure(figsize=(10, 6))
        
        # Helper to plot data and regression line
        def plot_series(data_dict, label, color):
            if task in data_dict:
                # Extract x (number of layers) and y (score)
                items = data_dict[task]
                x = [len(item['Layers']) for item in items]
                y = [item['Score'] for item in items]
                
                # Add initial point
                if items:
                    initial_score = items[0]['Score'] - items[0]['Diff']
                    x.insert(0, 0)
                    y.insert(0, initial_score)
                
                if x and y:
                    # Plot scatter points
                    plt.scatter(x, y, color=color, label=label, alpha=0.7)
                    
                    # Calculate and plot regression line if we have enough points
                    if len(x) > 1:
                        z = np.polyfit(x, y, 1)
                        p = np.poly1d(z)
                        # Create a range for the line to ensure it spans the data nicely
                        x_line = np.linspace(min(x), max(x), 100)
                        plt.plot(x_line, p(x_line), color=color, linestyle='--', alpha=0.5, label=f'{label} Trend')

        # Plot Baseline
        plot_series(baseline_data, 'Baseline prompt', 'blue')
        
        # Plot Positive
        plot_series(positive_data, 'Positive prompt', 'red')

        # Plot Negative
        plot_series(negative_data, 'Negative prompt', 'green')

        # Plot attacked
        plot_series(attacked_data, 'TextFooler prompt', 'purple')
        plot_series(attacked_dwb_data, 'DWB prompt', 'cyan')

        plt.title(f'Trend for Task: {task}')
        plt.xlabel('Number of Layers')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # Save plot
        # Replace spaces or special chars in filename if necessary
        safe_filename = "".join([c if c.isalnum() or c in ('-', '_') else "_" for c in task])
        save_path = os.path.join(output_dir, f'{safe_filename}_trend.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    main()