# GPT Debug Responses

GPT-generated error analysis and code fixes during Bayesian optimization.

## Structure

```
gpt_debug_responses/
└── debug_response_{timestamp}.json
```

Each JSON contains:
- `error_log`: Original error message from training
- `analysis`: GPT's error diagnosis
- `suggested_fix`: Code correction recommendations
- `confidence`: Fix confidence score
- `timestamp`: When error occurred

Files are automatically created by `error_monitor.py` when training errors are detected during BO runs.
