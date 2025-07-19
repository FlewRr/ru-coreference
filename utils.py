from allennlp.predictors.predictor import Predictor
# import allennlp_models.coref  # нужно для регистрации моделей
import sys

predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"
)

text = "Barack Obama was born in Hawaii. He was elected president in 2008."

result = predictor.predict(document=text)

clusters = result["clusters"]
tokens = result["document"]

print("Coreference Clusters:")
for cluster in clusters:
    spans = [" ".join(tokens[start:end + 1]) for start, end in cluster]
    print(" -", spans)
