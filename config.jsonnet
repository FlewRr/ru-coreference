local transformer_model = "DeepPavlov/rubert-base-cased";

{
  "dataset_reader": {
    "type": "coref",
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": transformer_model
    },
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": transformer_model
      }
    },
    "max_span_width": 10
  },

  "train_data_path": "train.jsonlines",
  "validation_data_path": "val.jsonlines",

  "model": {
    "type": "coref",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": transformer_model,
          "train_parameters": true
        }
      }
    },
    "feature_size": 150,
    "max_span_width": 10,
    "spans_per_word": 0.4,
    "lexical_dropout": 0.2,
    "antecedent_dropout": 0.2
  },

  "data_loader": {
    "batch_size": 2
  },

  "trainer": {
    "num_epochs": 10,
    "optimizer": {
      "type": "adamw",
      "lr": 1e-5
    },
    "grad_clipping": 1.0,
    "callbacks": [
    {
      "type": "wandb",
      "project": "ru-coreference",
      "group": "experiment-1",
      "run_name": "run1",
      "log_model": true,
      "save_artifacts": true
    }
  ]
  }
}
