import re
import sys
import pandas as pd
from matplotlib import pyplot as plt


def extract_data(log_file):
    with open(log_file, "r") as f:
        extracted_logs = {1: []}
        curr_epoch = 1
        for line in f.readlines():
            line = line.strip().replace("\x08", "")
            new_epoch_match = re.match(r'Epoch (\d+)/', line)
            if new_epoch_match:
                curr_epoch = int(new_epoch_match.group(1))
                extracted_logs[curr_epoch] = []
            else:
                data = line.split("-")[2:]
                metrics = {}
                for d in data:
                    name, num = d.split(":")
                    metrics[name.strip()] = float(num.strip())

                extracted_logs[curr_epoch].append(metrics)

    return (extracted_logs)


def save_data(data, metrics=["accuracy", "loss", "val_loss", "val_accuracy"]):
    # Dictionary to hold all y_values for each metric
    all_y_values = {metric: [] for metric in metrics}

    # Extracting y values for each metric
    for metric in metrics:
        for epoch in data.keys():
            for i, step in enumerate(data[epoch]):
                if not (metric in step.keys()):
                    continue
                if step == {}:
                    continue
                all_y_values[metric].append(step[metric])

    # Create a DataFrame with all y_values
    df = pd.DataFrame(all_y_values)

    # Write DataFrame to an Excel file
    excel_file_path = 'metrics_data.xlsx'
    df.to_excel(excel_file_path, index=False)


if __name__ == "__main__":
    logs_path = sys.argv[-1]
    save_data(extract_data(logs_path))
