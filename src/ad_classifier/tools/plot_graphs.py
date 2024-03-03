import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_inline.backend_inline import set_matplotlib_formats

set_matplotlib_formats('retina', quality=1000)


def plot_bar(df):
    # Extract necessary columns
    model_names = df['Model Name']
    accuracy = df['Accuracy']
    precision = df['Precision']
    sensitivity = df['Sensitivity']
    f1_score = df['F1 Score']

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))

    # Define colors
    colors = ['#1f77b4', '#4d88ff', '#99bbff', '#cceeff']
    metrics = ['Accuracy', 'Precision', 'Sensitivity', 'F1 Score']
    data = [accuracy, precision, sensitivity, f1_score]

    # Bar width
    bar_width = 0.2
    # Positions of bars on x-axis
    r = range(len(model_names))

    for i, metric in enumerate(metrics):
        bars = ax.bar([x + i * bar_width for x in r], data[i],
                      width=bar_width, label=metric, color=colors[i])

    ax.set_ylabel("%")
    ax.set_xticks([r + 1.5 * bar_width for r in range(len(model_names))])
    ax.set_xticklabels(model_names)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.spines['left'].set_color('#DDDDDD')
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)

    # Add labels
    for bar in ax.patches:
        ax.annotate("{:.1f}".format(bar.get_height()*100),
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha='center', va='bottom', color='black')

    # Add legend
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Add title
    plt.title('Evaluative Performance Metrics by Model', fontsize=16)

    # Adjust layout
    fig.tight_layout()
    plt.savefig("./var_graph.png")


def plot_line(df):
    # Extract necessary columns
    epochs = df['Epoch']
    validation_loss1 = df['SDG Optimisation Validation Loss']
    validation_loss2 = df['No Optimisation Validation Loss']
    validation_loss3 = df['ADAM']

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot lines for loss and validation loss
    ax.plot(epochs, validation_loss1, label='SDG Optimisation Validation Loss',
            marker='o', color='#ff7f0e')
    ax.plot(epochs, validation_loss2, label='No Optimisation Validation Loss',
            marker='o', color='#99bbff')
    ax.plot(epochs, validation_loss3, label='ADAM Optimisation Validation Loss',
            marker='o', color='#1f77b4')

    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.spines['left'].set_color('#DDDDDD')
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)

    # Add legend
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Add title
    plt.title('Loss of Models using SDG and No Optimisers', fontsize=16)

    # Adjust layout
    fig.tight_layout()
    plt.savefig("./line_graph.png")


def plot_conf():
    # Create a sample confusion matrix
    cm = np.array([[790, 3,  2,  5],
                   [7, 786,  5,  2],
                   [7, 10, 780,  3],
                   [2,  9,   2, 787]])

    # Define class labels
    classes = ["AD", "CN", "MCI", "pMCI"]

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, cmap=sns.color_palette(
        ['#1f77b4', '#4d88ff', '#99bbff', '#cceeff']), fmt='g', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Diagnosis')
    plt.ylabel('True Diagnosis')
    plt.title('Confusion Matrix of CogniNet', fontsize=16)
    plt.savefig("conf.png")


def plot_full_bars():
    data = {
        'Optimizer': ['ADAM', 'ADAM', 'ADAM', 'ADAM', 'SDG', 'SDG', 'SDG', 'SDG', 'None', 'None', 'None', 'None'],
        'Loss Function': ['Sparse', 'Sparse', 'Categorical', 'Categorical', 'Sparse', 'Sparse', 'Categorical', 'Categorical', 'Sparse', 'Sparse', 'Categorical', 'Categorical'],
        'Pooling Method': ['Average', 'Max', 'Average', 'Max', 'Average', 'Max', 'Average', 'Max', 'Average', 'Max', 'Average', 'Max'],
        'Accuracy': [92.375, 92.044, 91.813, 98.219, 75.969, 75.875, 74.719, 74.625, 91.688, 89.656, 94.156, 90.531],
        'Precision': [92.435, 92.044, 91.876, 98.224, 76.986, 76.333, 75.757, 76.438, 91.687, 89.725, 94.190, 90.644],
        'Sensitivity': [92.375, 91.969, 91.813, 98.219, 75.969, 75.875, 74.719, 74.625, 91.688, 89.656, 94.156, 90.531],
        'F1 Score': [0.92358853, 0.919861441, 0.918043505, 0.982192544, 0.75562, 0.75693, 0.74918, 0.74332, 0.91648, 0.89682, 0.94161, 0.90475]
    }

    df = pd.DataFrame(data)

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 8))

    # Define colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    metrics = ['Accuracy', 'Precision', 'Sensitivity', 'F1 Score']

    # Bar width
    bar_width = 0.2

    for i, metric in enumerate(metrics):
        for idx, row in df.iterrows():
            x = i * bar_width + idx + 0.5
            ax.bar(x, row[metric], width=bar_width,
                   label=row['Optimizer'], color=colors[i])

    # Add labels and title
    ax.set_ylabel("%")
    ax.set_xticks([i + 0.5 for i in range(len(df) * len(metrics))])
    ax.set_xticklabels([f"{row['Pooling Method']} ({row['Loss Function']})" for idx,
                        row in df.iterrows()], rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.spines['left'].set_color('#DDDDDD')
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(
        'Evaluative Performance Metrics by Optimizer, Loss Function, and Pooling Method', fontsize=16)

    # Adjust layout
    fig.tight_layout()
    plt.savefig("./graph.png")
