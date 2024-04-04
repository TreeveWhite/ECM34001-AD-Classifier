from matplotlib import rc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_inline.backend_inline import set_matplotlib_formats
import matplotlib as mpl

plt.style.use('seaborn-v0_8-darkgrid')
font = {'size': 16}
mpl.rc('font', **font)
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

set_matplotlib_formats('retina', quality=1000)


def plot_survey_bars():
    data = {
        'Question': [
            'How valuable do you think\nCogniCheck and similar AI\nsystems could be in Healthcare?',
            "How successfully do the attention map visualisations\nexplain reasoning behind a diagnosis?",
            'How easy is the system to use and navigate?',
            "As a whole, how do you rate the system?"
        ],
        'Average Response (1-10)': [8, 7.5, 9, 9.5]
    }

    # data = {
    #     'Question': [
    #         'Ease of use',
    #         'Clear diagnoses and results',
    #         'Understandable AI diagnoses',
    #         'Intuitive system',
    #         'Customisability and user profiles',
    #         'System Compatibility with tablets & mobile devices',
    #         'Collaborative features'
    #     ],
    #     'Average Response (1-10)': [8.75, 9.5, 8.5, 7, 4.5, 7, 7]
    # }

    df = pd.DataFrame(data)
   # Extract necessary columns
    questions = df['Question']
    ratings = df['Average Response (1-10)']

    # Plotting
    fig, ax = plt.subplots(figsize=(18, 6))

    # Define colors
    colors = ['#1f77b4', '#4d88ff', '#99bbff', "#ff7f0e"]

    # Use barh for horizontal bars
    ax.barh(questions, ratings, color=colors)

    ax.set_xlabel("Average Response (1-10)", fontsize=20)  # Change to xlabel
    ax.set_xlim(xmin=0, xmax=10)  # Change to xlim
    ax.grid(axis='x', linestyle='--', alpha=0.7)  # Change to x-axis grid

    # Adjust the font size of tick labels
    ax.tick_params(axis='y', which='major', labelsize=24)

    # Add labels
    for bar in ax.patches:
        ax.annotate("{:.1f}".format(bar.get_width()),
                    (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    va='center', ha='left', color='black', fontsize=20)  # Adjust position and alignment

    # Add title
    ax.set_title(
        'Results of End User Testing Survey', fontsize=24)

    # Adjust layout
    fig.tight_layout()
    fig.savefig("plotter.png", bbox_inches='tight', dpi=300)


def plot_bar(df):
    # Extract necessary columns
    model_names = df['Model Name']
    accuracy = df['Accuracy']
    precision = df['Precision']
    sensitivity = df['Sensitivity']
    f1_score = df['F1 Score']

    accuracy = list(map(lambda x: x * 100, accuracy))

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Define colors
    colors = ['#1f77b4', '#4d88ff', '#99bbff', '#cceeff', "#ff7f0e"]

    ax1.bar(model_names, accuracy, color=colors)

    ax1.set_ylabel("%")
    ax1.set_ylim(ymin=0, ymax=100)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Add labels
    for bar in ax1.patches:
        ax1.annotate("{:.3f}".format(bar.get_height()),
                     (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     ha='center', va='bottom', color='black')

    # Add legend
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Add title
    ax1.set_title(
        'Accuracy of Experimental Models during Testing', fontsize=16)

    # Create a table
    table_data = {
        'Model Name': model_names,
        'Sensitivity': sensitivity,
        'Precision': precision,
        'F1 Score': f1_score
    }
    ax2.axis('off')  # Hide axis
    print(table_data.values())
    ax2.table(cellText=list(table_data.values()),
              colLabels=list(table_data.keys()), loc='center')

    # Adjust layout
    fig.tight_layout()
    fig.savefig("plotter.png", bbox_inches='tight', dpi=300)


def plot_line(df):
    # Extract necessary columns
    epochs = df['Epoch']

    results1 = list(map(lambda x: x * 100, df['densenet_accuracy']))
    results2 = list(map(lambda x: x * 100, df['densenet_val_accuracy']))
    results3 = df['densenet_loss']
    results4 = df['densenet_val_loss']

    results5 = list(map(lambda x: x * 100, df['resnet_accuracy']))
    results6 = list(map(lambda x: x * 100, df['resnet_val_accuracy']))
    results7 = df['resnet_loss']
    results8 = df['resnet_val_loss']

    results9 = list(map(lambda x: x * 100, df['inceptionnet_accuracy']))
    results10 = list(map(lambda x: x * 100, df['inceptionnet_val_accuracy']))
    results11 = df['inceptionnet_loss']
    results12 = df['inceptionnet_val_loss']

    results13 = list(map(lambda x: x * 100, df['cogninet_accuracy']))
    results14 = list(map(lambda x: x * 100, df['cogninet_val_accuracy']))
    results15 = df['cogninet_loss']
    results16 = df['cogninet_val_loss']

    results17 = list(map(lambda x: x * 100, df['vggnet_accuracy']))
    results18 = list(map(lambda x: x * 100, df['vggnet_val_accuracy']))
    results19 = df['vggnet_loss']
    results20 = df['vggnet_val_loss']

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(epochs, results2, label='DenseNet201 Validation Accuracy',
            marker='o', color='#1f77b4')
    ax.plot(epochs, results6, label='ResNet50 Validation Accuracy',
            marker='o', color='#79B473')
    ax.plot(epochs, results10, label='InceptionNet V3 Validation Accuracy',
            marker='o', color='#EB5E55')
    # ax.plot(epochs, results15, label='CogniNet Training Loss',
    #         marker='o', color='#79B473')
    ax.plot(epochs, results18, label='VGGNet19 Validation Accuracy',
            marker='o', color='#ff7f0e')

    ax.set_ylabel("Validation Accuracy")
    # ax.set_ylim(ymin=0.8, ymax=1)
    # ax.set_xlim(xmin=1, xmax=10)
    ax.set_xlabel("Epoch")

    # Add legend
    ax.legend(loc="lower right")

    # Add title
    plt.title(
        'Validation Accuracy of Baseline Deep Learning Models', fontsize=16)

    # Adjust layout
    fig.tight_layout()
    fig.savefig("plotter.png", bbox_inches='tight', dpi=300)
    plt.show()


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

    # Convert data to DataFrame
    df = pd.DataFrame(data)

    # Pivot the DataFrame to create a matrix of accuracies
    heatmap_data = df.pivot_table(values='F1 Score', index='Optimizer', columns=[
        'Pooling Method', 'Loss Function'])

    # Plot the heatmap
    plt.figure()
    sns.heatmap(heatmap_data, cmap='RdYlGn',
                annot=True, fmt=".2f", linewidths=.5)

    plt.title('F1 Score Heatmap of Fine Tuning Experiments')
    plt.xlabel('Pooling Method - Loss Function')
    plt.ylabel('Optimiser')
    plt.savefig("full.png", bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    df = pd.read_excel(
        "/home/white/uni_workspace/ecm3401-dissertation/data/All Modesl Training.xlsx")
    plot_line(df)

    # plot_survey_bars()
