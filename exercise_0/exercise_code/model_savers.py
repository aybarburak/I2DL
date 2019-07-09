"""Helper Functions for Saving Model Files."""
import os
import pickle as pickle


def save_model(modelname, data):
    """Save given model with the given name."""
    directory = 'models'
    model = {modelname: data}
    if not os.path.exists(directory):
        os.makedirs(directory)
    pickle.dump(model, open(directory + '/' + modelname + '.p', 'wb'))


def save_test_model(test_model):
    """Wraps model saving for the softmax_classifier model."""
    modelname = 'test_model'
    save_model(modelname, test_model)
