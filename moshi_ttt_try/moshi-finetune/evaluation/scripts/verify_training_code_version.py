#!/usr/bin/env python3
"""
Check if the currently running training is using the old or new checkpoint code
"""

import sys
import os

sys.path.append('/home/alufr/ttt_tests/moshi')
sys.path.insert(0, '.')

def check_running_training_code():
    """Check what version of the code is actually loaded in Python"""
    print("🔍 CHECKING LOADED CODE VERSION")
    print("=" * 60)
    
    try:
        # Import the hybrid layer
        from moshi_ttt.hybrid_layer import HybridSeqModelingBlock
        import inspect
        
        # Get the source code of _apply_ttt_processing method
        source = inspect.getsource(HybridSeqModelingBlock._apply_ttt_processing)
        
        print("Current _apply_ttt_processing method in memory:")
        print("-" * 50)
        
        # Look for the checkpoint_group_size line
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'checkpoint_group_size' in line:
                print(f"Line {i+1}: {line.strip()}")
                
        # Check what the actual line contains
        if "checkpoint_group_size = 0" in source:
            print("\n✅ LOADED CODE: Uses checkpoint_group_size = 0 (FIXED)")
            return True
        elif "checkpoint_group_size = min(max(1, NC), NC)" in source:
            print("\n❌ LOADED CODE: Uses old checkpoint logic (NOT FIXED)")
            return False
        else:
            print("\n🤔 LOADED CODE: Unknown checkpoint logic")
            # Print the relevant section
            for i, line in enumerate(lines):
                if 'checkpoint_group_size' in line:
                    start = max(0, i-2)
                    end = min(len(lines), i+3)
                    print("Context:")
                    for j in range(start, end):
                        marker = ">>> " if j == i else "    "
                        print(f"{marker}{lines[j]}")
            return False
        
    except Exception as e:
        print(f"❌ Failed to check loaded code: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_if_training_needs_restart():
    """Check if the training process needs to be restarted to pick up the fix"""
    print(f"\n🔍 TRAINING RESTART ANALYSIS")
    print("-" * 50)
    
    # Check when the fix was applied vs when training started
    hybrid_layer_path = '/home/alufr/ttt_tests/moshi-finetune/moshi_ttt/hybrid_layer.py'
    
    import os
    import time
    
    # Get file modification time
    mod_time = os.path.getmtime(hybrid_layer_path)
    mod_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mod_time))
    
    print(f"hybrid_layer.py last modified: {mod_time_str}")
    
    # Check if any python processes are running training
    import subprocess
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        python_processes = []
        
        for line in result.stdout.split('\n'):
            if 'python' in line and ('train' in line or 'moshi' in line):
                python_processes.append(line)
        
        if python_processes:
            print(f"\nFound {len(python_processes)} potential training processes:")
            for proc in python_processes:
                print(f"   {proc}")
            
            print(f"\n🚨 DIAGNOSIS:")
            print(f"   If these processes started BEFORE {mod_time_str},")
            print(f"   they are using the OLD code and need to be restarted!")
        else:
            print(f"\nNo Python training processes found")
            
    except Exception as e:
        print(f"Could not check running processes: {e}")

def main():
    print("🧪 TRAINING CODE VERSION CHECK")
    print("=" * 60)
    
    # Check what's loaded in memory
    loaded_code_is_fixed = check_running_training_code()
    
    # Check if training needs restart
    check_if_training_needs_restart()
    
    print(f"\n🎯 SUMMARY:")
    print("-" * 40)
    
    if loaded_code_is_fixed:
        print("✅ Python has the fixed code loaded")
        print("   → The problem is NOT the checkpoint fix")
        print("   → Need to look elsewhere for TTT parameter freeze")
    else:
        print("❌ Python has the OLD code loaded")
        print("   → The training process started BEFORE the fix was applied")
        print("   → SOLUTION: Restart the training process to pick up the fix")
    
    return loaded_code_is_fixed

if __name__ == "__main__":
    success = main()
    
    if not success:
        print(f"\n🚀 ACTION REQUIRED:")
        print(f"   1. Stop the current training process")
        print(f"   2. Restart training to pick up the checkpoint fix")
        print(f"   3. Monitor logs for changing ttt_alpha values")
    else:
        print(f"\n🔍 NEXT STEPS:")
        print(f"   The checkpoint fix is loaded but TTT still frozen")
        print(f"   Need to investigate other causes of parameter freeze")