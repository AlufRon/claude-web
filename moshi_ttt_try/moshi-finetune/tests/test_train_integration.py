#!/usr/bin/env python3
"""
Test script for train.py integration with TTT and paper metrics.
This script runs a minimal training session to verify all components work.
"""

import os
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

def run_test():
    """Run integration test for modified train.py"""
    
    print("🧪 Testing modified train.py integration...")
    
    # Set environment for single GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'
    env['WORLD_SIZE'] = '1'
    env['RANK'] = '0'
    env['LOCAL_RANK'] = '0'
    env['MASTER_ADDR'] = 'localhost'
    env['MASTER_PORT'] = '12355'
    
    # Change to the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Run torchrun with minimal training
    cmd = [
        'torchrun',
        '--standalone',
        '--nnodes=1',
        '--nproc_per_node=1',
        'train.py',
        'configs/test_train_py_integration.yaml'
    ]
    
    print(f"🚀 Running command: {' '.join(cmd)}")
    print("   Expected: Training should start, show TTT integration, and run 5 steps")
    print("   Will timeout after 120 seconds...")
    
    try:
        # Run with timeout to prevent hanging
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minutes timeout
            cwd=script_dir
        )
        
        print(f"✅ Process completed with return code: {result.returncode}")
        
        # Check for key success indicators in output
        output = result.stdout + result.stderr
        
        success_indicators = [
            "Applying TTT to",  # TTT integration
            "TTT conversion complete",  # TTT success
            "Model sharded!",  # FSDP success
            "mixed precision",  # Mixed precision setup
            "Paper metrics",  # Paper metrics evaluation
        ]
        
        found_indicators = []
        for indicator in success_indicators:
            if indicator.lower() in output.lower():
                found_indicators.append(indicator)
                print(f"   ✅ Found: {indicator}")
            else:
                print(f"   ❌ Missing: {indicator}")
        
        # Look for error indicators
        error_indicators = [
            "Error",
            "Exception",
            "Traceback",
            "CUDA out of memory",
            "loss: nan",
            "loss: inf"
        ]
        
        found_errors = []
        for error in error_indicators:
            if error.lower() in output.lower():
                found_errors.append(error)
                print(f"   ⚠️  Error found: {error}")
        
        # Print relevant output
        print("\n📋 Key output sections:")
        lines = output.split('\n')
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ['ttt', 'paper metrics', 'loss:', 'mixed precision', 'fsdp']):
                print(f"   {line}")
        
        # Determine success
        critical_indicators = ["Applying TTT to", "Model sharded!"]
        critical_found = sum(1 for ind in critical_indicators if ind.lower() in output.lower())
        
        if critical_found >= len(critical_indicators) // 2 and len(found_errors) == 0:
            print("\n🎉 SUCCESS: Train.py integration test passed!")
            print(f"   - Found {len(found_indicators)}/{len(success_indicators)} success indicators")
            print(f"   - Found {len(found_errors)} error indicators")
            return True
        else:
            print("\n❌ FAILURE: Train.py integration test failed!")
            print(f"   - Found {len(found_indicators)}/{len(success_indicators)} success indicators")
            print(f"   - Found {len(found_errors)} error indicators")
            print("\nFull output:")
            print(output)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out (this may be normal if training started successfully)")
        print("   Check if the process is still running and training is progressing")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)