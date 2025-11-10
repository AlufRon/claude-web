"""
Simple test to verify our debug logging infrastructure works
"""

import sys
sys.path.append('/home/alufr/ttt_tests/moshi-finetune')

from moshi_ttt.model_utils import save_ttt_states, restore_ttt_states
import torch
import torch.nn as nn

class MockTTTLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(10, 10))
        
    def save_ttt_states(self):
        return {'W1': self.W1.clone().detach()}
        
    def restore_ttt_states(self, state):
        if 'W1' in state:
            with torch.no_grad():
                self.W1.copy_(state['W1'])

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = MockTTTLayer()

def test_debug_logging():
    print("🔍 Testing debug logging infrastructure...")
    
    model = MockModel()
    
    # Test save
    print("🔍 Testing save_ttt_states...")
    saved = save_ttt_states(model)
    
    # Test restore
    print("🔍 Testing restore_ttt_states...")
    result = restore_ttt_states(model, saved)
    
    print(f"🔍 Result: {result}")
    print("🔍 Debug logging test completed")

if __name__ == "__main__":
    test_debug_logging()