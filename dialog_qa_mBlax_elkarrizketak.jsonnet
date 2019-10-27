{
    "dataset_reader": {
        "type": "quac",
        "lazy": true,
        "num_context_answers": 2,
        "token_indexers": {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": "mBlax",
                "do_lowercase": true,
                "use_starting_offsets": true,
                "truncate_long_sequences": false                             
            },
            "token_characters": {
                "type": "characters",
                "character_tokenizer": {
                    "byte_encoding": "utf-8",
                    "end_tokens": [
                        260
                    ],
                    "start_tokens": [
                        259
                    ]
                },
                "min_padding_length": 3
            }
        }
    },
    "iterator": {
        "type": "bucket",
        "batch_size": 10,
        "max_instances_in_memory": 1000,
        "sorting_keys": [
            [
                "question",
                "num_fields"
            ],
            [
                "passage",
                "num_tokens"
            ]
        ]
    },
    "model": {
        "type": "dialog_qa",
        "dropout": 0.2,
        "initializer": [],
        "marker_embedding_dim": 10,
        "num_context_answers": 2,
        "phrase_layer": {
            "type": "gru",
            "bidirectional": true,
            "hidden_size": 100,
            "input_size": 916,  // BERT (768) + cnn (100) + num_context_answers (2) * marker_embedding_dim (10)
            "num_layers": 1
        },
        "residual_encoder": {
            "type": "gru",
            "bidirectional": true,
            "hidden_size": 100,
            "input_size": 200,
            "num_layers": 1
        },
        "span_end_encoder": {
            "type": "gru",
            "bidirectional": true,
            "hidden_size": 100,
            "input_size": 400,
            "num_layers": 1
        },
        "span_start_encoder": {
            "type": "gru",
            "bidirectional": true,
            "hidden_size": 100,
            "input_size": 200,
            "num_layers": 1
        },
        "text_field_embedder": {
            "allow_unmatched_keys": true,
            "embedder_to_indexer_map": {
                "bert": ["bert", "bert-offsets"],
                "token_characters": ["token_characters"]
            },
            "token_embedders": {
                "bert": {
                    "type": "bert-pretrained",
                    "pretrained_model": "Blax"
                },
                "token_characters": {
                    "type": "character_encoding",
                    "embedding": {
                        "embedding_dim": 16,
                        "num_embeddings": 262
                    },
                    "encoder": {
                        "type": "cnn",
                        "embedding_dim": 16,
                        "num_filters": 128,
                        "ngram_filter_sizes": [3],
                        "conv_layer_activation": "relu"
                    }
                }
            }
        }

    },
    "train_data_path": "elkarrizketak.quac.train.json",
    "validation_data_path": "elkarrizketak.quac.dev.json",
    "trainer": {
        "cuda_device": 0,
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "mode": "max",
            "patience": 3
        },
        "num_epochs": 30,
        "optimizer": {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9
        },
        "patience": 10,
        "validation_metric": "+f1"
    },
    "validation_iterator": {
        "type": "bucket",
        "batch_size": 3,
        "max_instances_in_memory": 1000,
        "sorting_keys": [
            [
                "question",
                "num_fields"
            ],
            [
                "passage",
                "num_tokens"
            ]
        ]
    }
}
