from mijn_model import model_factory

model = model_factory()
preds = model.predict("data-studenten.csv")
print(preds[:10])
