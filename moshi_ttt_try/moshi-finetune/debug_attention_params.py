"""Debug script to check attention parameter names in TTT-integrated model."""
import torch
from moshi.models import loaders
from finetune.args import TrainArgs, TTTArgs, DataArgs, OptimizerArgs
from finetune.ttt_integration import apply_ttt_to_model

# Create minimal args
args = TrainArgs(
    data=DataArgs(train_data='dummy', eval_data=''),
    run_dir='/tmp/debug',
    optim=OptimizerArgs(lr=1e-4),
    ttt=TTTArgs(
        enable=True,
        layers="1",
        unfrozen_attention_layers="1",
    ),
    lora=None,
    full_finetuning=False,
)

# Load model
print("Loading model...")
checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
    hf_repo='kyutai/moshiko-pytorch-bf16',
)
model = checkpoint_info.get_lm_model(device='cpu')

# Apply TTT
print("\nApplying TTT to layer 1...")
apply_ttt_to_model(model, args)

# Check parameter names
print("\nChecking parameter names in layer 1:")
print("="*80)
for name, param in model.named_parameters():
    if 'layers.1.' in name and 'attn' in name.lower():
        print(f"{name}")
        print(f"  Shape: {param.shape}")
        print(f"  Numel: {param.numel():,}")
        print()

print("\nChecking all layer 1 parameter patterns:")
print("="*80)
layer_1_params = [name for name, _ in model.named_parameters() if 'layers.1.' in name]
for name in layer_1_params[:10]:  # First 10
    print(f"  {name}")
if len(layer_1_params) > 10:
    print(f"  ... and {len(layer_1_params) - 10} more")
