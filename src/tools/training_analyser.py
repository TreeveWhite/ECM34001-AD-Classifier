import re

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


def plot(data, metric="accuracy"):
    x_values = []
    y_values = []
    for epoch in data.keys():
        for i, step in enumerate(data[epoch]):
            if step == {}:
                continue
            x_values.append(f"{epoch}.{i+1}")
            y_values.append(step[metric])

    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')
    plt.title(metric)
    plt.xlabel('Epoch.Step')
    plt.ylabel('Accuracy')
    plt.savefig(f"{metric}.png")
    plt.show()


if __name__ == "__main__":
    logs_path = "/home/white/uni_workspace/ECM34001-AD-Classifier/models/DENSE_NET-2024-02-06-19:11/log.log"
    plot(extract_data(logs_path))
