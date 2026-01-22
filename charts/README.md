# Charts Directory

Auto-generated Bayesian optimization visualization outputs.

## Structure

```
charts/
├── {Timestamp}_BO_{ModelName}/
│   ├── convergence_plot.png       # BO convergence curve
│   ├── parameter_importance.png   # Hyperparameter importance
│   ├── training_curves.png        # Training/validation curves
│   └── bo_summary.txt            # Text summary
└── pipeline_summary_{Timestamp}.json  # Pipeline execution summary
```

Charts are automatically created by `visualization.py` during model training.
