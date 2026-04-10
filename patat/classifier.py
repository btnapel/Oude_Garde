from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class StudentenClassifier:
    def __init__(self):
        self.threshold = 0.45
        self.target_col = "prognose10jaar"
        self.drop_cols = ["Unnamed: 0", "Individu-ID"]
        self._train_path = Path(__file__).resolve().parent / "train.csv"
        self.pipeline = None
        self._fit()

    def _basic_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace({"?": np.nan, "nan": np.nan, "None": np.nan, "": np.nan})
        return df

    def _coerce_numeric_like_columns(self, X: pd.DataFrame):
        X = X.copy()
        categorical_cols = [c for c in X.columns if X[c].dtype == object]
        numeric_cols = [c for c in X.columns if c not in categorical_cols]

        for col in categorical_cols[:]:
            coerced = pd.to_numeric(X[col], errors="coerce")
            # Als bijna alles numeriek is, behandel de kolom als numeriek.
            if coerced.notna().mean() >= 0.80:
                X[col] = coerced
                categorical_cols.remove(col)
                numeric_cols.append(col)

        return X, numeric_cols, categorical_cols

    def _build_pipeline(self, X: pd.DataFrame):
        X, numeric_cols, categorical_cols = self._coerce_numeric_like_columns(X)

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
                    numeric_cols,
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_cols,
                ),
            ],
            remainder="drop",
        )

        model = GradientBoostingClassifier(
            random_state=42,
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        return pipeline, X

    def _fit(self):
        df = pd.read_csv(self._train_path)
        df = self._basic_clean(df)

        y = (df[self.target_col] == "CHD+").astype(int).to_numpy()
        X = df.drop(columns=[self.target_col], errors="ignore")
        X = X.drop(columns=[c for c in self.drop_cols if c in X.columns], errors="ignore")

        self.pipeline, X = self._build_pipeline(X)
        self.pipeline.fit(X, y)

    def predict(self, filename):
        df = pd.read_csv(filename)
        df = self._basic_clean(df)
        X = df.drop(columns=[self.target_col], errors="ignore")
        X = X.drop(columns=[c for c in self.drop_cols if c in X.columns], errors="ignore")

        # Zelfde conversie-logica als tijdens trainen.
        X, _, _ = self._coerce_numeric_like_columns(X)

        proba = self.pipeline.predict_proba(X)[:, 1]
        preds = np.where(proba >= self.threshold, "CHD+", "CHD-")
        return np.asarray(preds)
