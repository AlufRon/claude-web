"""
Simple test to replicate the backward graph error without complex TTT operations.

This demonstrates the core issue: calling init_weights() during training
destroys the computation graph needed for backpropagation.
"""

import torch
import torch.nn as nn


class SimpleTTTLayer(nn.Module):
    """Simplified TTT layer that demonstrates the backward graph issue"""
    
    def __init__(self, dim=512):
        super().__init__()
        # These represent TTT's learnable parameters W1, b1, W2, b2
        self.W1 = nn.Parameter(torch.randn(dim, dim * 4))
        self.b1 = nn.Parameter(torch.zeros(dim * 4))
        self.W2 = nn.Parameter(torch.randn(dim * 4, dim))
        self.b2 = nn.Parameter(torch.zeros(dim))
        
    def init_weights(self):
        """This is what breaks the computation graph"""
        nn.init.normal_(self.W1, mean=0.0, std=0.02)
        nn.init.zeros_(self.b1)
        nn.init.normal_(self.W2, mean=0.0, std=0.02)
        nn.init.zeros_(self.b2)
        
    def forward(self, x):
        # Simple MLP forward that uses the parameters
        h = torch.matmul(x, self.W1) + self.b1
        h = torch.relu(h)
        output = torch.matmul(h, self.W2) + self.b2
        return output


def test_backward_graph_error():
    """
    Test the exact sequence that causes the error:
    1. Forward pass (builds computation graph)
    2. init_weights() (destroys computation graph)
    3. backward() (fails because graph is destroyed)
    """
    
    print("🔍 Testing backward graph error with simple TTT layer...")
    
    # Create layer and input
    layer = SimpleTTTLayer(dim=128)
    layer.train()
    
    x = torch.randn(2, 64, 128, requires_grad=True)  # [batch, seq, dim]
    
    print(f"✅ Setup: Layer with {sum(p.numel() for p in layer.parameters())} parameters")
    
    # Step 1: Forward pass that builds computation graph
    print("📈 Step 1: Forward pass (builds computation graph)")
    output = layer(x)
    loss = output.mean()  # Simple loss
    
    print(f"   Output shape: {output.shape}")
    print(f"   Loss: {loss.item():.6f}")
    print(f"   Computation graph exists: {loss.grad_fn is not None}")
    
    # Step 2: Call init_weights() - this is what our reset function does
    print("🔄 Step 2: Call init_weights() (destroys computation graph)")
    
    # Store parameter values before reset
    W1_before = layer.W1.clone()
    
    # This is the problematic call that destroys the computation graph
    layer.init_weights()
    
    # Verify parameters changed
    W1_after = layer.W1.clone()
    params_changed = not torch.allclose(W1_before, W1_after, atol=1e-6)
    print(f"   Parameters changed: {params_changed}")
    
    # Step 3: Try backward - this should fail
    print("⬅️  Step 3: Call backward() - expecting failure...")
    
    try:
        loss.backward()
        print("❌ UNEXPECTED: backward() succeeded when it should have failed!")
        return False
    except RuntimeError as e:
        error_msg = str(e)
        if "Trying to backward through the graph a second time" in error_msg or \
           "saved tensor" in error_msg or "freed" in error_msg:
            print(f"✅ SUCCESS: Replicated the backward graph error!")
            print(f"   Error: {error_msg}")
            return True
        else:
            print(f"❓ Different error: {error_msg}")
            return False


def test_correct_approach():
    """
    Test the fix: manual parameter reset with torch.no_grad()
    """
    
    print("\n🔧 Testing correct approach with torch.no_grad()...")
    
    # Create layer and input  
    layer = SimpleTTTLayer(dim=128)
    layer.train()
    
    x = torch.randn(2, 64, 128, requires_grad=True)
    
    # Step 1: Forward pass
    print("📈 Step 1: Forward pass")
    output = layer(x)
    loss = output.mean()
    
    print(f"   Loss: {loss.item():.6f}")
    
    # Step 2: Reset parameters safely with no_grad()
    print("🔄 Step 2: Reset parameters with torch.no_grad()")
    
    # Store values before reset
    W1_before = layer.W1.clone()
    
    # CORRECT approach: manual reset with no_grad()
    with torch.no_grad():
        nn.init.normal_(layer.W1, mean=0.0, std=0.02)
        nn.init.zeros_(layer.b1)
        nn.init.normal_(layer.W2, mean=0.0, std=0.02)
        nn.init.zeros_(layer.b2)
    
    # Verify parameters changed
    W1_after = layer.W1.clone()
    params_changed = not torch.allclose(W1_before, W1_after, atol=1e-6)
    print(f"   Parameters changed: {params_changed}")
    
    # Step 3: Backward should work now
    print("⬅️  Step 3: Call backward() - should work...")
    
    try:
        loss.backward()
        print("✅ SUCCESS: backward() worked correctly!")
        print(f"   Input gradients computed: {x.grad is not None}")
        return True
    except RuntimeError as e:
        print(f"❌ ERROR: backward() failed: {str(e)}")
        return False


def test_no_grad_wrapper_approach():
    """
    Test wrapping init_weights() call with no_grad() - this should also work
    """
    
    print("\n🛠️  Testing no_grad() wrapper around init_weights()...")
    
    layer = SimpleTTTLayer(dim=128)
    layer.train()
    
    x = torch.randn(2, 64, 128, requires_grad=True)
    
    # Forward pass
    print("📈 Step 1: Forward pass")
    output = layer(x)
    loss = output.mean()
    print(f"   Loss: {loss.item():.6f}")
    
    # Reset with no_grad() wrapper
    print("🔄 Step 2: Call init_weights() wrapped in torch.no_grad()")
    
    W1_before = layer.W1.clone()
    
    # This should work: wrap the problematic call in no_grad()
    with torch.no_grad():
        layer.init_weights()
    
    W1_after = layer.W1.clone() 
    params_changed = not torch.allclose(W1_before, W1_after, atol=1e-6)
    print(f"   Parameters changed: {params_changed}")
    
    # Try backward
    print("⬅️  Step 3: Call backward() - should work...")
    
    try:
        loss.backward()
        print("✅ SUCCESS: backward() worked with no_grad() wrapper!")
        return True
    except RuntimeError as e:
        print(f"❌ ERROR: backward() failed: {str(e)}")
        return False


if __name__ == "__main__":
    print("=" * 80)
    print("SIMPLE BACKWARD GRAPH ERROR TEST")
    print("=" * 80)
    
    # Test 1: Replicate the error
    error_replicated = test_backward_graph_error()
    
    # Test 2: Manual parameter reset with no_grad()
    manual_reset_works = test_correct_approach()
    
    # Test 3: Wrapper approach
    wrapper_works = test_no_grad_wrapper_approach()
    
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(f"✅ Error replicated: {error_replicated}")
    print(f"✅ Manual reset works: {manual_reset_works}")
    print(f"✅ Wrapper approach works: {wrapper_works}")
    
    if error_replicated and (manual_reset_works or wrapper_works):
        print("\n🎯 CONCLUSION:")
        print("   PROBLEM: Calling init_weights() during training destroys computation graph")
        print("   SOLUTION: Wrap init_weights() call in torch.no_grad()")
        print("   This prevents the parameter reassignment from affecting the computation graph")
    else:
        print("\n❌ Test inconclusive - need further investigation")
    
    print("=" * 80)