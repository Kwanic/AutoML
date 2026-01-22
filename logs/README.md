# Execution Logs

Timestamped pipeline execution logs.

## Structure

```
logs/
└── YYYY-MM-DD_HH-MM-SS.log
```

**Format:** `YYYY-MM-DD HH:MM:SS - LEVEL - MODULE - MESSAGE`

**Levels:**
- INFO: Normal operations and progress
- WARNING: Non-critical issues
- ERROR: Failures with stack traces
- DEBUG: Detailed diagnostics (disabled by default)

Logs are automatically created by `logging_config.py` for each pipeline run.
