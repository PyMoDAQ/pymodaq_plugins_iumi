from tensorflow.keras.models import load_model
from pathlib import Path
import numpy as np

# TODO : Implement a way for the user to chose what model to load. 
# TODO : Implement some utility function that adapts the code the expected shape from the neural network. (For example)

class PinemAnalysis : 
    """
    Class to use the neural network and perform the data analysis in live.

    Args:
    model_file : str
        Path to the model file (hdf5 format)

    It can be any keras-usable model file.
    """
    def __init__(self,model_file) : 
        self.model = load_model(model_file, compile = False)
        print('Loaded model from', model_file)

    # TODO : Implement scaling along with the normalization
        
    def eval_background(self, data) :
        """
        Evaluate the background of the spectrum.

        Args:
        data : np.ndarray
            Data to evaluate the background it should be a 1D array.
        """
        first10 = data[:int(data.shape[0] * 0.1)]
        last10 = data[int(data.shape[0] * 0.9):]
        return np.concatenate((first10, last10)).mean()

    def normalize(self, data) :
        """
        Normalize the data between 0 and 1.

        Args:
        data : np.ndarray
            Data to normalize it should be a 1D array.
        """ 
        M = np.max(data)
        m = np.min(data)
        return (data-m)/(M-m)
    
    def predict(self, data, remove_background = True) : 
        """
        Predict the shape of the spectrum (i.e. its underlying parameters) using the neural network.

        Args:
        data : np.ndarray
            Data to predict it should be a 1D array.
        """
        if remove_background :
            data = data - self.eval_background(data).clip(min=0)
        ndata = self.normalize(data)
        cdata = correct_center_of_mass(ndata)
        cdata = cdata[np.newaxis, :, np.newaxis]
        g = self.model.predict(cdata)
        return g
    
def center_of_mass(data : np.ndarray) :
    """
    Compute the center of mass of a collection of 1D array

    Parameters
    ----------
    data : 2D array
        The data to compute the center of mass with shape (number of points in the spectra,)
    """ 
    coords = np.arange(data.shape[0])
    com = np.sum(data*coords)/np.sum(data)
    return com


def eval_center_of_mass(data) :
    """
    Evaluate the center of mass of a set of spectra. The spectra should have the shape (number of spectra, spectra length).

    Args:
    - data (np.ndarray): The set of spectra to evaluate the center of mass of.
    """
    cdata = data.copy()
    hM = cdata.max()/1.3
    mask = cdata < hM
    cdata[mask] = 0
    return center_of_mass(cdata)
    
def correct_center_of_mass(data) :
    """
    Correct the center of mass of a set of spectra. The spectra should have the shape (number of spectra, spectra length).

    Args:
    - data (np.ndarray): The set of spectra to correct the center of mass of.
    """
    com = eval_center_of_mass(data)
    out  = np.roll(data, int(np.round(data.shape[0]/2 - com)))
    return out