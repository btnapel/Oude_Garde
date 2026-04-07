import pandas as pd
import numpy as np

class DemoClassifier:
    def predict(self, filename="competition_test.csv"):
        data = pd.read_csv(filename)
        return np.random.choice([True, False], size=(data.shape[0],))