# FOFE Character Encoding

## Structure of project

This project's first aim is to implement a neural layer in Pytorch which performs the FOFE method on character level described in [Zhang et al. (2015)](http://www.aclweb.org/anthology/P15-2081) to embed to the words. This layer is then passed to a bidirectional GRU architecture.
In a second step the new FOFE layer is compared to a classical, randomly initialised embedding layer.
The two architectures are tested on the English [ATIS dataset](https://github.com/Microsoft/CNTK/tree/master/Examples/LanguageUnderstanding/ATIS/Data) and on parts of the German [Tiger Corpus](http://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/TIGERCorpus/download/start.html).

## Structure of repository

The _src_ folder includes python and bash scripts designed for the different configurations. There is a main python script _tagger.py_ which uses the _fofe_model.py_ or _classic_model.py_ depending on the model to be trained. Data preparation for both corpora is done in _prep.py_.
To test different parameter configurations there is a wrapper class for the tagger module which can be used for hyper paramter optimisation in _hyperopt.py_.

### Implementation

#### Settings of Neural Network
– size of embedding layer = 50 (only for classic model)
– drop-out rate = 0.5
– size of hidden layer in GRU = 50
– optimiser = Adam with default learning rate (lr = 0.001) and no weight
decay
– loss function = cross entropy loss

### Results

| Data set | train loss | | dev loss | | test loss | | accuracy  || weighted F1 ||
|   | FOFE  | Classic | FOFE  | Classic  | FOFE  | Classic  | FOFE  | Classic  | FOFE  | Classic  |
|---|---|---|---|---|---|---|---|---|---|---|
| Atis| 0.28   | 0.04  | 0.34  | 0.08  | 0.47 | 0.19 | 0.91 | 0.98 | 0.48 | 0.74 |
|  Tiger |  0.94 |   0.11| 0.92  |  0.38 | 0.99 | 0.49 |0.71 | 0.91 | 0.5 | 0.78|
|---|---|---|---|---|---|---|---|---|---|---|

### ATIS corpus:

The Classic model clearly outperforms the FOFE model with a maximal accuracy of *0.98* in epoch 20 versus a maximal accuracy of *0.91* in epoch 30. 

- an arbitrarily chosen parameter setting reaches accuracy = 0.1, weighted F1 score = 0.14 (dev loss: 3.83) see _logs/screen-logs/fofe_encoding/log_atis_fofe.txt_
- hyperparameter optimisation: is not yet finished, but best configuration so far reaches accuracy = 0.07, weighted F1 score = 0.11 (best dev loss: 3.94), see _logs/screen-logs/fofe_encoding/log_hyper_atis_fofe_1.txt_

#### Classic Embedding

- an arbitrarily chosen parameter setting reaches accuracy = 0.09, weighted F1 score = 0.13 (dev loss: 3.9), see _logs/screen-logs/classic_embedding/log_atis_classic_1.txt_
- hyperparameter optimisation reaches accuracy = 0.09, weighted F1 score = 0.13 (best dev loss: 3.81), see _logs/screen-logs/classic_embedding/log_hyper_atis_classic.txt_

### Tiger corpus:

#### FOFE Encoding

- still awaiting results

#### Classic Embedding

- an arbitrarily chosen parameter setting reaches accuracy = 0.41, weighted F1 score = 0.42 (dev loss: 3.85), see _logs/screen-logs/classic_embedding/log_tiger_classic.txt_
- Comment: Looking at the log, the model clearly overfits, the numbers are taken implying early stopping

## Future work

- Rerun hyperparameter optimisation on all four configurations
- Implement early stopping (at least for Tiger corpus)
- Visualise results
