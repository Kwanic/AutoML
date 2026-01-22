# Trained Models

Saved PyTorch model checkpoints from successful training runs.

## Structure

```
trained_models/
└── {ModelName}_{Dataset}_acc{XX}_{Timestamp}.pth
```

**Examples:**
- `BiLSTM_Attention_ECG_acc87_20250102.pth`
- `QuantizedCNN_8bit_acc82_20250102.pth`

Models are automatically saved during BO trials and final training when performance improves.

**Loading:**
```python
import torch
model.load_state_dict(torch.load('trained_models/model.pth'))
model.eval()
```

**Note:** Large `.pth` files should not be committed to git (see `.gitignore`).
