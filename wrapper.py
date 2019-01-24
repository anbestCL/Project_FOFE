from tagger_cuda import Tagger


class Wrapper:

    def __init__(self, modelname, datafile, batch_size):
        self.modelname = modelname
        self.datafile = datafile
        self.batchsize = batch_size

    # config_dict={param: value_choice, ....}
    def learn(self, num_epochs, config_dict, seed):
        model = Tagger(self.modelname, self.datafile,
                       num_epochs, self.batchsize, **config_dict)
        # train
        metrics = model.train(num_epochs, seed, **config_dict)
        # metrics is dict = {epoch: (model, train_loss, dev_loss,test_loss, acc, f1_macro, f1_weighted)}
        return metrics


if __name__ == '__main__':
    learner = Wrapper('FOFE', 'Atis.json', 8)
    config_dict = {'embedding_size': 100, 'hidden_size': 100,
                   'dropout_rate': 0.5, 'learn_rate': 0.01, 'reg_factor': 0.0}
    learner.learn([0, 1, 5, 10], config_dict, 0)
