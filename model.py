"""
model om voorspellingen te doen


Author: Bert ten Napel, Maartje van der Hulst
Date: 7-4-2026
"""


import pandas as pd
import numpy as np

class DemoClassifier:
    """verhaal lalal"""

    def __init__(self, filename="data-studenten.csv"):
        """
        sllal
        """
        self.filename = filename
        self.data = self.dataprep()

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
        data = pd.read_csv(self.filename, sep=",", header=0, na_values="?",
                           true_values=["+"], false_values=["-"])
        for i in ("hypertensie", "hartinfarct", "diabetes", "nierziekte"):
            data[i].replace({"+", 1}, {"-", 0}, inplace=True)
        # print(data["prognose10jaar"].head())
        return data


    def predict(self):
        data = self.dataprep()
        return np.random.choice([True, False], size=(data.shape[0],))



DemoClassifier()

