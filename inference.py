from allennlp.predictors.predictor import Predictor
from allennlp_models.coref import coref
import sys

def load_model(model_path: str):
    return Predictor.from_path(model_path)

def predict_coref(predictor, text: str):
    result = predictor.predict(document=text)
    clusters = result["clusters"]
    tokens = result["document"]

    text_clusters = []
    for cluster in clusters:
        mentions = [" ".join(tokens[start:end+1]) for start, end in cluster]
        text_clusters.append(mentions)
    return text_clusters

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to trained model (model.tar.gz)")
    parser.add_argument("text_file", help="Path to .txt file")

    args = parser.parse_args()

    with open(args.text_file, "r", encoding="utf-8") as f:
        text = f.read()

    predictor = load_model(args.model_path)
    clusters = predict_coref(predictor, text)

    print("Coreference clusters:")
    for cluster in clusters:
        print(" -", cluster)
