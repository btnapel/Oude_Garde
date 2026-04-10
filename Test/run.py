from mijn_model import model_factory

model = model_factory()

comparison = model.compare(
    "data-competitie-noclass.csv",
    "y_true.npy"
)

scores = model.compare_scores(
    "data-competitie-noclass.csv",
    "y_true.npy"
)

print(comparison)
print("\nScores:")
print(scores)
