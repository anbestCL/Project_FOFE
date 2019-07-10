from tagger import Tagger


class Wrapper:

    """Wrapper for model to pass to hyper optimisation

    Arguments:
        modelname {string} - either "FOFE" for Fofe character encoding or "Classic" for classic trainable embedding layer
        datafile {string} - path to data
        paramfile {string} - path to save model and metrics to
        batch_size {number} - size of training batches

    Returns:
        dictionary -- maps each evaluation epoch to tuple of model, train_loss, dev_loss, test_loss,
                        accuracy, macro and weighted F1 score
                        hyper optimisation script chooses best config based on this dictionary
    """

    def __init__(self, modelname, datafile, paramfile, batch_size):
        self.modelname = modelname
        self.datafile = datafile
        self.batchsize = batch_size
        self.paramfile = paramfile

    def learn(self, num_epochs, config_dict, seed):
        # config_dict contains a chosen value for each parameter
        model = Tagger(self.modelname, self.datafile, self.paramfile,
                       num_epochs, self.batchsize, **config_dict)
        # train
        metrics = model.train(num_epochs, seed, **config_dict)
        # metrics is dict = {epoch: (model, train_loss, dev_loss,test_loss, acc, f1_macro, f1_weighted)}
        return metrics


if __name__ == '__main__':
    learner = Wrapper('FOFE', 'Atis.json', "hyper", 8)
    config_dict = {'embedding_size': 100, 'hidden_size': 100,
                   'dropout_rate': 0.5, 'learn_rate': 0.01, 'reg_factor': 0.0}
    learner.learn([0, 1, 5, 10], config_dict, 0)
