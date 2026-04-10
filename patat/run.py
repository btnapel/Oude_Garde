from model import model_factory

model = model_factory()

preds = model.predict("test.csv")

print(preds)