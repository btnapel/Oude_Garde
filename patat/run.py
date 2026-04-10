import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Model:
    def __init__(self, train_file="data-studenten.csv"):
        self.train_file = train_file
        self.target_col = None
        self.drop_cols = []
        self.num_cols = []
        self.cat_cols = []
        self.threshold = 0.5
        self.pipeline = None

        df = pd.read_csv(train_file)
        df = self._clean_df(df)

        self.target_col = df.columns[-1]

        # gooi kolommen weg die meestal niks helpen
        candidate_drop = ["Unnamed: 0", "Individu-ID"]
        self.drop_cols = [c for c in candidate_drop if c in df.columns]

        X = df.drop(columns=[self.target_col] + self.drop_cols, errors="ignore")
        y = df[self.target_col].astype(str)

        # numeriek proberen te maken waar dat logisch is
        X = self._coerce_numeric_columns(X)

        self.num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        self.cat_cols = [c for c in X.columns if c not in self.num_cols]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    self.num_cols,
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    self.cat_cols,
                ),
            ]
        )

        # ensemble voor betere robuustheid
        ensemble = VotingClassifier(
            estimators=[
                (
                    "lr",
                    LogisticRegression(
                        max_iter=4000,
                        class_weight="balanced",
                        C=0.5,
                        random_state=42,
                    ),
                ),
                (
                    "et",
                    ExtraTreesClassifier(
                        n_estimators=700,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ],
            voting="soft",
            weights=[3, 2, 1],
        )

        self.pipeline = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", ensemble),
            ]
        )

        # threshold tunen op trainset via cross-val
        self.threshold = self._find_best_threshold(X, y)

        # daarna fitten op alle train data
        self.pipeline.fit(X, y)

    def _clean_df(self, df):
        df = df.copy()
        df = df.replace("?", np.nan)
        df = df.replace("", np.nan)
        return df

    def _coerce_numeric_columns(self, X):
        X = X.copy()
        for col in X.columns:
            if X[col].dtype == object:
                try:
                    converted = pd.to_numeric(X[col])
                    # alleen overschrijven als dit echt zinvol is
                    if converted.notna().sum() >= max(3, int(0.7 * len(converted.dropna()))):
                        X[col] = converted
                except:
                    pass
        return X

    def _prepare_test_X(self, filename):
        df = pd.read_csv(filename)
        df = self._clean_df(df)

        if self.target_col in df.columns:
            df = df.drop(columns=[self.target_col], errors="ignore")

        df = df.drop(columns=self.drop_cols, errors="ignore")
        df = self._coerce_numeric_columns(df)

        return df

    def _find_best_threshold(self, X, y):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # classes staan alfabetisch: CHD+ eerst, CHD- daarna
        probs = cross_val_predict(
            clone(self.pipeline),
            X,
            y,
            cv=cv,
            method="predict_proba",
            n_jobs=None,
        )[:, 0]

        best_threshold = 0.5
        best_f1 = -1

        for threshold in np.linspace(0.10, 0.90, 161):
            preds = np.where(probs >= threshold, "CHD+", "CHD-")
            score = f1_score(y, preds, pos_label="CHD+")
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)

        return best_threshold

    def predict(self, filename):
        X = self._prepare_test_X(filename)
        probs = self.pipeline.predict_proba(X)[:, 0]  # kans op CHD+
        preds = np.where(probs >= self.threshold, "CHD+", "CHD-")
        return np.array(preds)

    def predict_proba_chdplus(self, filename):
        X = self._prepare_test_X(filename)
        probs = self.pipeline.predict_proba(X)[:, 0]
        return probs

    def score(self, filename):
        df = pd.read_csv(filename)
        df = self._clean_df(df)

        if self.target_col not in df.columns:
            raise ValueError("Geen echte labels gevonden in dit bestand.")

        y_true = df[self.target_col].astype(str)
        y_pred = self.predict(filename)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, pos_label="CHD+")

        return {
            "accuracy": acc,
            "f1": f1,
            "threshold": self.threshold,
        }


if __name__ == "__main__":
    model = Model("data-studenten.csv")

    print("Beste threshold:", model.threshold)

    # score op trainbestand zelf
    scores = model.score("data-studenten.csv")
    print("Accuracy:", scores["accuracy"])
    print("F1:", scores["f1"])

    preds = model.predict("data-studenten.csv")
    print("Eerste 20 voorspellingen:")
    print(preds[:20])