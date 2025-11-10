#!/usr/bin/env python3
"""
Calculate fair LoRA rank for TTT vs LoRA comparison
"""

def calculate_lora_params(rank, d_model=1024, num_layers=32):
    """Calculate LoRA parameter count
    
    LoRA adds A and B matrices for each linear layer:
    - Q, K, V, O matrices in attention (4 per layer)
    - FFN matrices (2 per layer) 
    - Total: 6 matrices per layer
    
    Each matrix pair adds: d_model * rank * 2 parameters
    """
    matrices_per_layer = 6  # Q, K, V, O + 2 FFN matrices
    params_per_matrix_pair = d_model * rank * 2
    params_per_layer = matrices_per_layer * params_per_matrix_pair
    total_params = params_per_layer * num_layers
    return total_params

def find_matching_rank(target_params, d_model=1024, num_layers=32):
    """Find LoRA rank that gives closest parameter count to target"""
    best_rank = 1
    best_diff = float('inf')
    
    for rank in range(1, 65):
        lora_params = calculate_lora_params(rank, d_model, num_layers)
        diff = abs(lora_params - target_params)
        if diff < best_diff:
            best_diff = diff
            best_rank = rank
            
    return best_rank, calculate_lora_params(best_rank, d_model, num_layers)

if __name__ == "__main__":
    print("🧮 LoRA vs TTT Fair Comparison Calculator")
    print("=" * 50)
    
    # Current configurations
    ttt_params = 4_354_080  # 4.3M (from logs)
    current_lora_rank = 64
    current_lora_params = calculate_lora_params(current_lora_rank)
    
    print(f"📊 Current Setup:")
    print(f"   TTT (1 layer): {ttt_params:,} parameters")
    print(f"   LoRA (rank {current_lora_rank}): {current_lora_params:,} parameters")
    print(f"   Ratio: LoRA has {current_lora_params/ttt_params:.1f}x more parameters")
    
    print(f"\n🎯 Fair Comparison Options:")
    
    # Option 1: Match TTT parameter count
    fair_rank, fair_lora_params = find_matching_rank(ttt_params)
    print(f"\n   Option 1 - Match TTT parameters:")
    print(f"   LoRA rank {fair_rank}: {fair_lora_params:,} parameters")
    print(f"   Difference: {abs(fair_lora_params - ttt_params):,} parameters")
    
    # Option 2: Compromise ranks
    for compromise_rank in [8, 12, 16, 20]:
        compromise_params = calculate_lora_params(compromise_rank)
        ratio = compromise_params / ttt_params
        print(f"\n   Option 2 - LoRA rank {compromise_rank}:")
        print(f"   LoRA: {compromise_params:,} parameters ({ratio:.1f}x TTT)")
    
    print(f"\n💡 Recommendation:")
    print(f"   Use LoRA rank {fair_rank} for exact parameter match")
    print(f"   Or LoRA rank 8-12 for close approximate match")