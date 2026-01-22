# Data Directory

Dataset storage for training and evaluation.

## Structure

```
data/
├── dataset1/           # Example: MIT-BIH ECG dataset
│   ├── X.npy          # Features
│   ├── y.npy          # Labels
│   └── README.md      # Dataset description
└── your_dataset/      # Add your own datasets here
```

## Supported Formats

- NumPy arrays (`.npy`, `.npz`)
- CSV files
- Images
- Time series sequences
- Custom data formats

The pipeline automatically detects and converts data using the Universal Data Converter from [adapters/](../adapters/).

Place your datasets in subdirectories here for automatic processing.
