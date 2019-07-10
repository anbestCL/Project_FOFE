import matplotlib.pyplot as plot
import pickle

histories_atis = {}
with open("params_atis_FOFE_metrics.txt", "rb") as f1:
    histories_atis["FOFE"] = pickle.load(f1)
histories_atis["FOFE"] = {i: (train, dev, test, acc, f1_inv_weighted) for i, (
    _, train, dev, test, acc, f1_inv_weighted, f1_weighted) in histories_atis["FOFE"].items()}
with open("params_atis_Classic_metrics.txt", "rb") as f2:
    histories_atis["Classic"] = pickle.load(f2)
histories_atis["Classic"] = {i: (train, dev, test, acc, f1_inv_weighted) for i, (
    _, train, dev, test, acc, f1_inv_weighted, f1_weighted) in histories_atis["Classic"].items()}


def plot_histories(histories):
    metrics_named = ['training loss', 'development loss',
                     'test loss', 'accuracy', 'inversely-weighted F1 score']
    for i, metric in enumerate(metrics_named):
        plot.subplot(2, 3, i+1)
        for model_type, metrics in histories.items():
            train_loss = [values[i] for j, values in metrics.items()]
            plot.plot(metrics.keys(), train_loss, label=model_type)
            plot.title(metric)
            plot.xlabel("Epoch")
        plot.legend()
    plot.tight_layout()
    plot.show()


plot_histories(histories_atis)


def max_values(histories, index):
    max_ = 4 if index < 3 else 0
    max_it = 0

    for i, values in histories.items():
        if index < 3:
            if values[index] < max_:
                max_ = values[index]
                max_it = i
        else:
            if values[index] > max_:
                max_ = values[index]
                max_it = i
    return max_, max_it


for model in ['FOFE', 'Classic']:
    for index in [0, 1, 2, 3, 4]:
        f_max, f_it = max_values(histories_atis[model], index)
        print(f_max, f_it)


histories_tiger = {}
with open("params_tiger_FOFE_metrics.txt", "rb") as f1:
    histories_tiger["FOFE"] = pickle.load(f1)
histories_tiger["FOFE"] = {i: (train, dev, test, acc, f1_inv_weighted) for i, (
    _, train, dev, test, acc, f1_inv_weighted, f1_weighted) in histories_tiger["FOFE"].items()}
with open("params_tiger_Classic_metrics.txt", "rb") as f2:
    histories_tiger["Classic"] = pickle.load(f2)
histories_tiger["Classic"] = {i: (train, dev, test, acc, f1_inv_weighted) for i, (
    _, train, dev, test, acc, f1_inv_weighted, f1_weighted) in histories_tiger["Classic"].items()}

plot_histories(histories=histories_tiger)

for model in ['FOFE', 'Classic']:
    for index in [0, 1, 2, 3, 4]:
        f_max, f_it = max_values(histories_tiger[model], index)
        print(f_max, f_it)
