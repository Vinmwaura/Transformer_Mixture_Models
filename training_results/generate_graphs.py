import os
import csv
import matplotlib.pyplot as plt

def main():
    folder_path = os.path.dirname(os.path.abspath(__file__))

    # Comment/Uncomment to plot the graphs.
    loss_csv_files = [
        # "Wikipedia(num_mixture=1,mixture_type=models).csv",
        "Wikipedia(num_mixture=3,mixture_type=models).csv",
        # "Wikipedia(num_mixture=1,num_blocks=1,mixture_type=blocks).csv",
        # "Wikipedia(num_mixture=3,num_blocks=1,mixture_type=blocks).csv",
        # "Wikipedia(num_mixture=5,num_blocks=1,mixture_type=blocks).csv",
        # "Wikipedia(num_mixture=8,num_blocks=1,mixture_type=blocks).csv",
        "Wikipedia(num_mixture=16,num_blocks=1,mixture_type=blocks).csv",
        # "Wikipedia(num_mixture=1,num_blocks=2,mixture_type=blocks).csv",
        # "Wikipedia(num_mixture=2,num_blocks=2,mixture_type=blocks).csv",
        # "Wikipedia(num_mixture=8,num_blocks=2,mixture_type=blocks).csv",
        "Wikipedia(num_mixture=16,num_blocks=2,mixture_type=blocks).csv"
    ]

    # Create plots.
    plt.figure(figsize=(10, 6))
    
    for loss_csv_file in loss_csv_files:
        label = loss_csv_file.split(".")[0]

        loss_fpath = os.path.join(
            folder_path,
            "Loss",
            loss_csv_file)
        with open(loss_fpath, "r") as f:
            csv_reader = csv.reader(f)

            test_loss = []
            global_steps = []

            for row in csv_reader:
                global_steps.append(int(row[0]))
                test_loss.append(float(row[1]))
    
        plt.plot(global_steps, test_loss, label=label, marker='o')

    # Add title and labels.
    plt.title('Comparison of Testing Loss over time.')
    plt.xlabel('Global Step')
    plt.ylabel('Test Loss')

    # Add grid.
    plt.grid(True)

    # Add legend.
    plt.legend()

    # Show plot.
    plt.show()

if __name__ == "__main__":
    main()
