#!/usr/bin/env python3
"""
INVESTIGATE TTT OOM: Detailed analysis of why TTT hits OOM
"""

import sys
import torch
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def analyze_error_patterns():
    """Analyze the different error patterns we've seen"""
    print("🔍 TTT OOM ERROR ANALYSIS")
    print("=" * 80)
    
    print("📊 MEMORY USAGE COMPARISON:")
    
    print("\n1. VANILLA MOSHI + LORA (batch=16):")
    print("   ✅ Peak memory: 36.7GB")
    print("   ✅ Current memory: 22.3GB")
    print("   ✅ Status: SUCCESS - completes training step")
    print("   ✅ Additional params: ~33M LoRA parameters")
    
    print("\n2. TTT 4 LAYERS (batch=1):")
    print("   ❌ Memory usage: 47.28GB")
    print("   ❌ Error location: mb_loss.backward() - backward pass")
    print("   ❌ TTT params: 17M parameters")
    
    print("\n3. TTT 1 LAYER (batch=16):")
    print("   ❌ Memory usage: 47.08GB") 
    print("   ❌ Error location: forward pass - gating_forward_kernel")
    print("   ❌ TTT params: 4.3M parameters")
    
    print("\n4. TTT 1 LAYER (batch=1):")
    print("   ❌ Memory usage: 47.38GB")
    print("   ❌ Error location: backward pass - TTT mini-batch processing")
    print("   ❌ Error details: empty_strided_cuda((32, 512, 128))")
    
    print("\n🎯 KEY OBSERVATIONS:")
    print("1. Even 1 TTT layer with batch=1 uses 47.38GB")
    print("2. Vanilla + 33M LoRA params uses only 36.7GB")
    print("3. TTT 4.3M params somehow uses 10GB+ more memory")
    print("4. The memory issue is NOT just parameter count!")

def analyze_ttt_memory_overhead():
    """Analyze TTT's memory overhead beyond parameters"""
    print("\n" + "=" * 80)
    print("TTT MEMORY OVERHEAD ANALYSIS")
    print("=" * 80)
    
    print("🔍 TTT vs LoRA Memory Comparison:")
    print("\nLoRA Memory Usage:")
    print("   - Parameters: 33M")
    print("   - Gradients: 33M (same size as params)")
    print("   - Optimizer states: 66M (Adam: momentum + variance)")
    print("   - Total extra memory: ~132M parameters worth")
    print("   - Result: Uses 36.7GB total")
    
    print("\nTTT Memory Usage:")
    print("   - Parameters: 4.3M (much smaller than LoRA)")
    print("   - BUT: TTT does inner-loop optimization during forward/backward")
    print("   - Inner gradients: Must compute gradients for TTT layers")
    print("   - Activations: Must store activations for multiple mini-batches")
    print("   - Format conversions: Moshi ↔ TTT tensor reshaping")
    print("   - Compilation cache: torch.compile creates additional memory")
    print("   - Result: Uses 47.38GB total (+10GB+ overhead!)")

def analyze_specific_error_location():
    """Analyze the specific error location"""
    print("\n" + "=" * 80)
    print("SPECIFIC ERROR LOCATION ANALYSIS")
    print("=" * 80)
    
    print("💥 Latest Error Details:")
    print("Location: /tmp/torchinductor_alufr/vu/cvu7s4z5lf43h3c7drddump5c34b47wkywum3nfeizvcy4ipivtv.py")
    print("Line: buf38 = empty_strided_cuda((32, 512, 128), (65536, 128, 1), torch.bfloat16)")
    print("Memory requested: 20.00 MiB")
    print("Available memory: 4.44 MiB")
    print("Total in use: 47.38 GiB")
    
    print("\n🔍 Error Analysis:")
    print("1. This is inside a torch-compiled function")
    print("2. Trying to allocate tensor of shape (32, 512, 128)")
    print("3. Size: 32 × 512 × 128 = 2,097,152 elements")
    print("4. Memory: 2M elements × 2 bytes (bfloat16) = 4MB")
    print("5. But error says trying to allocate 20MB - why?")
    
    print("\n🎯 Key Insight:")
    print("The issue is NOT the 20MB allocation itself.")
    print("The issue is that we're already at 47.38GB/47.40GB (99.95% full)")
    print("ANY allocation fails when memory is completely full!")

def analyze_ttt_inner_loop_memory():
    """Analyze TTT's inner loop memory requirements"""
    print("\n" + "=" * 80)
    print("TTT INNER LOOP MEMORY ANALYSIS")
    print("=" * 80)
    
    print("🔄 TTT Processing Steps:")
    print("1. Forward pass: Normal transformer computation")
    print("2. Format conversion: Moshi → TTT format")
    print("3. TTT mini-batch processing: MULTIPLE inner forward/backward steps")
    print("4. Gradient computation: Compute gradients for TTT parameters")
    print("5. Parameter updates: Update TTT parameters")
    print("6. Format conversion: TTT → Moshi format")
    print("7. Continue with rest of model")
    
    print("\n💾 Memory Requirements for Each Step:")
    print("Step 2: Format conversion creates additional tensor copies")
    print("Step 3: Mini-batch processing needs:")
    print("   - Input tensors for each mini-batch")
    print("   - Intermediate activations for each mini-batch") 
    print("   - Gradient tensors for each mini-batch")
    print("   - This happens INSIDE the forward pass!")
    
    print("\n🎯 The Memory Explosion:")
    print("TTT doesn't just add 4.3M parameters.")
    print("It adds MULTIPLE COPIES of activations and gradients")
    print("because it does gradient descent DURING forward pass!")
    
    print("\nExample: For mini_batch_size=2:")
    print("   - Original tensor: [batch, seq, 4096]")
    print("   - Mini-batch 1 processing: needs activations + gradients")
    print("   - Mini-batch 2 processing: needs MORE activations + gradients")
    print("   - All stored simultaneously for backward pass")
    print("   - Memory multiplier: potentially 4-8x per TTT layer!")

def calculate_memory_scaling():
    """Calculate expected memory scaling"""
    print("\n" + "=" * 80)
    print("MEMORY SCALING CALCULATION")
    print("=" * 80)
    
    print("📊 Memory Scaling Analysis:")
    
    # Base model memory
    base_memory_gb = 36.7  # Vanilla Moshi with LoRA
    
    print(f"Base model memory: {base_memory_gb}GB")
    
    # TTT memory overhead
    ttt_overhead_observed = 47.38 - base_memory_gb
    print(f"TTT overhead (observed): {ttt_overhead_observed:.1f}GB")
    
    # Parameter comparison
    lora_params = 33_000_000  # 33M parameters
    ttt_params = 4_354_080    # 4.3M parameters
    
    print(f"LoRA parameters: {lora_params:,}")
    print(f"TTT parameters: {ttt_params:,}")
    print(f"Parameter ratio: TTT has {ttt_params/lora_params:.2f}x LoRA parameters")
    
    # But memory usage comparison
    print(f"Memory ratio: TTT uses {ttt_overhead_observed/0:.2f}x more memory than LoRA")
    
    print("\n🎯 Conclusion:")
    print(f"TTT with {ttt_params:,} parameters uses MORE memory")
    print(f"than LoRA with {lora_params:,} parameters!")
    print("This confirms TTT has massive memory overhead beyond just parameters.")

def suggest_solutions():
    """Suggest potential solutions"""
    print("\n" + "=" * 80)
    print("POTENTIAL SOLUTIONS")
    print("=" * 80)
    
    print("🚀 Immediate Solutions:")
    print("1. DISABLE TORCH.COMPILE")
    print("   - Remove @torch.compile decorators from TTT code")
    print("   - Compilation creates additional memory overhead")
    print("   - May reduce memory usage significantly")
    
    print("\n2. REDUCE TTT MINI-BATCH SIZE")
    print("   - Current: mini_batch_size=2")
    print("   - Try: mini_batch_size=1")
    print("   - This reduces inner-loop memory requirements")
    
    print("\n3. GRADIENT CHECKPOINTING FOR TTT")
    print("   - Don't store all TTT activations")
    print("   - Recompute during backward pass")
    print("   - Trade computation for memory")
    
    print("\n4. SEQUENTIAL TTT PROCESSING")
    print("   - Process TTT mini-batches one at a time")
    print("   - Don't store all simultaneously")
    print("   - Clear intermediate tensors aggressively")
    
    print("\n🎯 Hardware Solutions:")
    print("1. Use larger GPU (80GB A100/H100)")
    print("2. Use multi-GPU with model parallelism")
    print("3. Use CPU offloading for TTT computations")
    
    print("\n📋 Next Steps:")
    print("1. Try disabling torch.compile first")
    print("2. Reduce TTT mini-batch size to 1")
    print("3. If still OOM, TTT may not be feasible on 48GB GPU")

def main():
    """Run the investigation"""
    analyze_error_patterns()
    analyze_ttt_memory_overhead()
    analyze_specific_error_location()
    analyze_ttt_inner_loop_memory()
    calculate_memory_scaling()
    suggest_solutions()

if __name__ == "__main__":
    main()