from allennlp_models.coref.models.coref import CoreferenceResolver
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data import Vocabulary


def build_model(vocab: Vocabulary):
    transformer_model = "DeepPavlov/rubert-base-cased"

    embedder = PretrainedTransformerEmbedder(
        model_name=transformer_model,
        train_parameters=True,
    )

    text_field_embedder = BasicTextFieldEmbedder({
        "tokens": embedder
    })

    model = CoreferenceResolver(
        vocab=vocab,
        text_field_embedder=text_field_embedder,
        contextualizer=None,
        feature_size=150,
        max_span_width=10,
        spans_per_word=0.4,
        lexical_dropout=0.2,
        antecedent_dropout=0.2,
        initializer=None
    )

    return model
