# FOFE Character Encoding

## Structure of project

This project's first aim is to implement a neural layer in Pytorch which performs the FOFE character encoding described in [Zhang et al. (2015)](http://www.aclweb.org/anthology/P15-2081) to embed to the words. This layer is then passed to a bidirectional GRU architecture.
In a second step the FOFE encoding is compared to a classical embedding layer which is initialised randomly.
The two architectures are tested on the English [ATIS dataset](https://github.com/Microsoft/CNTK/tree/master/Examples/LanguageUnderstanding/ATIS/Data) and on a part of the German [Tiger Corpus](http://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/TIGERCorpus/download/start.html).

## Structure of repository

The _src_ folder includes python and bash scripts designed for the different configurations. There is a main python script _tagger.py_ which uses the _fofe_model.py_ or _classic_model.py_ depending on the model to be trained. Data preparation for both corpora is done in _prep.class_.
To test different parameter configurations there is a wrapper class for the tagger module which is used for hyper paramter optimisation in _hyperopt.py_

## Preliminary results

### ATIS corpus:

#### FOFE Encoding

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
