# FOFE Character Encoding

## Project

This project's first aim is to implement a neural layer in Pytorch which performs the FOFE method on character level described in [Zhang et al. (2015)](http://www.aclweb.org/anthology/P15-2081) to embed to the words. This layer is then passed to a bidirectional GRU architecture.
In a second step the new FOFE layer is compared to a classical, randomly initialised embedding layer.
The two architectures are tested on the English [ATIS dataset](https://github.com/Microsoft/CNTK/tree/master/Examples/LanguageUnderstanding/ATIS/Data) and on parts of the German [Tiger Corpus](http://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/TIGERCorpus/download/start.html).

## Repository

The [source](src) folder includes python and bash scripts designed for the different configurations. There is a main [tagger](src/py-scripts/tagger.py) program which uses the [FOFE](src/py-scripts/fofe_model.py)_ or the [Classic](src/py-scripts/classic_model.py) depending on the model to be trained. [Data preparation](src/py-scripts/prep.py) for both corpora is done in advance.
To test different parameter configurations there is a wrapper class for the tagger module which can be used for [hyper paramter optimisation](src/py-scripts/hyperopt.py).

### Implementation

#### Settings of Neural Network
* size of embedding layer = 50 (only for classic model)
* drop-out rate = 0.5
* size of hidden layer in GRU = 50
* optimiser = Adam with default learning rate (lr = 0.001) and no weight
decay
* loss function = cross entropy loss

### Results

Data set | train loss | | dev loss | | test loss | | accuracy  || weighted F1 | forgetting factor
-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-------
Atis/Tiger | FOFE  | Classic | FOFE  | Classic  | FOFE  | Classic  | FOFE  | Classic  | FOFE  | Classic  |  
 Atis| 0.28   | 0.04  | 0.34  | 0.08  | 0.47 | 0.19 | 0.91 | 0.98 | 0.48 | 0.74 | 0.98
 Tiger |  0.94 |   0.11| 0.92  |  0.38 | 0.99 | 0.49 |0.71 | 0.91 | 0.5 | 0.78 | 1.27
 
More details including visualisations can be found in the written [report](documentation.pdf).

### Conclusion

From the results obtained from two different data sets, using the FOFE method as an alternative embedding layer for tagging tasks does not lead to an increase in performance. It might be that different parameter settings produce better results. This could be efficiently tested using hyper parameter optimisation.

## Future work

- Rerun hyperparameter optimisation on all four configurations
- Implement early stopping (at least for Tiger corpus)
