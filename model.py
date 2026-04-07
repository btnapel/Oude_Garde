import pandas as pd
import numpy as np

class DemoClassifier:
    """verhaal lalal"""

    def __init__(self, filename="competition_test.csv"):
        """
        sllal
        """
        self.filename = filename

        return 0

    def __repr__(self):
        """
        string representation for debugging purposes
        :return: string
        """

        return f"stringverhaallala{self.filename}"

    def dataprep(self):
        """
        dataprep dingen
        :return:
        """
        data = pd.read_csv(self.filename, sep=",", header=0)
        return data


    def predict(self):
        data = self.dataprep()
        return np.random.choice([True, False], size=(data.shape[0],))