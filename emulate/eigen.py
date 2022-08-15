

class EigenEmulator:
    R"""An eigen-emulator.

    Parameters
    ----------
    H0 :
        The operator that does not depend on the parameters
    H1 :
        The operator that depends linearly on the parameters

    """

    def __init__(self, H0, H1):
        self.H0 = H0
        self.H1 = H1

    def fit(self, p_train):
        R"""Fit the emulator at the training points

        Parameters
        ----------
        p_train
            The training parameters.

        Returns
        -------
        self
        """
        pass

    def predict(self, p):
        pass
