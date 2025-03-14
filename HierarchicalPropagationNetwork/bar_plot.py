import os
import matplotlib.pyplot as plt
import numpy as np


def parse_metrics_file(metrics_file):
    metrics = {}
    with open(metrics_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            key, value = line.strip().split(' : ')
            metrics[key.strip()] = float(value.strip())
    return metrics


def plot_metrics_barplot(metrics_directory, save_path):
    metrics_files = [file for file in os.listdir(metrics_directory) if file.endswith('_metrics.txt')]
    metrics_data = {metric: [] for metric in ['Accuracy', 'Precision', 'Recall', 'F1']}
    classifiers = []

    for metrics_file in metrics_files:
        classifier_name = metrics_file.split('_metrics.txt')[0]
        classifiers.append(classifier_name)
        metrics = parse_metrics_file(os.path.join(metrics_directory, metrics_file))
        for metric, value in metrics.items():
            metrics_data[metric].append(value)

    labels = list(metrics_data.keys())
    values = list(metrics_data.values())
    num_metrics = len(labels)
    num_classifiers = len(classifiers)

    colors = plt.cm.tab10.colors[:num_classifiers]

    fig, ax = plt.subplots(num_metrics, 1, figsize=(10, 8))
    fig.suptitle('Metrics Comparison', fontsize=16)

    for i in range(num_metrics):
        for j in range(num_classifiers):
            ax[i].bar(j, values[i][j], color=colors[j], label=classifiers[j])
        ax[i].set_ylabel(labels[i])
        ax[i].set_xticks(range(num_classifiers))
        ax[i].set_xticklabels([])
        ax[i].set_ylim(0, 1)
        ax[i].set_yticks(np.arange(0, 1.1, 0.1))
        ax[i].grid(axis='y')

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Salva il plot come immagine
    plt.savefig(save_path)
    #plt.show()


# Directory contenente i file metrics
metrics_directory = 'data/metrics/politifact_metrics'
# Percorso di salvataggio dell'immagine
save_path = 'data/bar_plots/politifact.png'
plot_metrics_barplot(metrics_directory, save_path)

# Directory contenente i file metrics
metrics_directory2 = 'data/metrics/gossipcop_metrics'
# Percorso di salvataggio dell'immagine
save_path2 = 'data/bar_plots/gossipcop.png'
plot_metrics_barplot(metrics_directory2, save_path2)

