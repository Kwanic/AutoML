# Generated Training Functions

AI-generated PyTorch training code storage with hyperparameter search spaces.

## Structure

```
generated_training_functions/
└── training_function_{data_type}_{model_name}_{timestamp}.json
```

Each JSON contains:
- `training_code`: Complete executable PyTorch function
- `bo_config`: Hyperparameter search space (types, ranges, defaults)
- `confidence`: GPT confidence score (0-1)
- `data_profile`: Original data characteristics
- `metadata`: Generation timestamp and model info

Files are automatically created by `ai_code_generator.py` during model generation.
