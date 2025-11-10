"""Simple debug to check if attention params match pattern."""

# Test the pattern matching
test_names = [
    "transformer.layers.1.self_attn.in_proj_weight",
    "transformer.layers.1.self_attn.out_proj.weight",
    "transformer.layers.1.original_layer.self_attn.in_proj_weight",
    "transformer.layers.1.seq_modeling_block.self_attn.in_proj_weight",
    "transformer.layers.1.norm1.weight",
]

def is_attention_parameter_in_layer(param_name: str, layer_indices: list) -> bool:
    """Check if parameter is attention parameter in specified layers.

    Handles both direct and wrapped layer patterns:
    - Direct: transformer.layers.1.self_attn.in_proj_weight
    - Wrapped: transformer.layers.1.original_layer.self_attn.in_proj_weight
    """
    if not layer_indices:
        return False
    for layer_idx in layer_indices:
        # Check both direct and wrapped patterns
        direct_pattern = f"transformer.layers.{layer_idx}.self_attn."
        wrapped_pattern = f"transformer.layers.{layer_idx}.original_layer.self_attn."
        if direct_pattern in param_name or wrapped_pattern in param_name:
            return True
    return False

print("Testing pattern matching:")
print("="*80)
for name in test_names:
    result = is_attention_parameter_in_layer(name, [1])
    print(f"{result:5} | {name}")
