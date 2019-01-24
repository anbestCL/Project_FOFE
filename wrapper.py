from tagger_cuda import Tagger


class Wrapper:

    def __init__(self, modelname, datafile):
        self.modelname = modelname
        self.datafile = datafile

    # config_dict={param: value_choice, ....}
    def learn(self, num_epochs, config_dict, seed):
        model = Tagger(self.modelname, self.datafile,
                       num_epochs, config_dict)
        # train
        metrics = model.train(num_epochs, config_dict)

        return metric(=accuracy)


if __name__ == '__main__':
    learner = Wrapper('FOFE', 'Atis.json')
