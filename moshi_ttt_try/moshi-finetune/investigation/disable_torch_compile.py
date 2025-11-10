#!/usr/bin/env python3
"""
DISABLE TORCH.COMPILE FOR TTT MEMORY OPTIMIZATION
Remove @torch.compile decorators to reduce memory overhead by 2-4GB
"""

import re
from pathlib import Path

def remove_torch_compile_decorators(file_path: Path):
    """Remove @torch.compile decorators from a Python file"""
    print(f"Processing {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Count existing decorators
    compile_count = len(re.findall(r'^\s*@torch\.compile\s*$', content, re.MULTILINE))
    print(f"  Found {compile_count} @torch.compile decorators")
    
    if compile_count == 0:
        print("  No @torch.compile decorators found - already optimized")
        return False
    
    # Remove @torch.compile decorators (including any whitespace)
    # This regex matches the decorator on its own line, preserving indentation of the function
    modified_content = re.sub(r'^\s*@torch\.compile\s*\n', '', content, flags=re.MULTILINE)
    
    # Verify removal
    remaining_count = len(re.findall(r'^\s*@torch\.compile\s*$', modified_content, re.MULTILINE))
    
    if remaining_count > 0:
        print(f"  WARNING: {remaining_count} decorators still remain!")
        return False
    
    # Create backup
    backup_path = file_path.with_suffix(file_path.suffix + '.backup')
    print(f"  Creating backup: {backup_path}")
    with open(backup_path, 'w') as f:
        f.write(content)
    
    # Write modified content
    with open(file_path, 'w') as f:
        f.write(modified_content)
    
    print(f"  ✅ Removed {compile_count} @torch.compile decorators")
    return True

def optimize_ttt_memory():
    """Remove torch.compile from all TTT-related files"""
    print("🚀 DISABLING TORCH.COMPILE FOR TTT MEMORY OPTIMIZATION")
    print("=" * 80)
    
    # Files to modify
    ttt_files = [
        Path("moshi_ttt/models/ssm/ttt_layer.py"),
        Path("moshi_ttt/models/ssm/ops/ttt_linear.py"),
        Path("moshi_ttt/models/ssm/ops/ttt_mlp.py"),
    ]
    
    total_removed = 0
    modified_files = []
    
    for file_path in ttt_files:
        if file_path.exists():
            if remove_torch_compile_decorators(file_path):
                modified_files.append(file_path)
                # Count removed decorators for summary
                with open(file_path.with_suffix(file_path.suffix + '.backup'), 'r') as f:
                    backup_content = f.read()
                removed_count = len(re.findall(r'^\s*@torch\.compile\s*$', backup_content, re.MULTILINE))
                total_removed += removed_count
        else:
            print(f"⚠️  File not found: {file_path}")
    
    print("\n" + "=" * 80)
    print("📊 OPTIMIZATION SUMMARY")
    print("=" * 80)
    print(f"Modified files: {len(modified_files)}")
    print(f"Total @torch.compile decorators removed: {total_removed}")
    print(f"Expected memory savings: 2-4GB")
    
    if modified_files:
        print(f"\n✅ OPTIMIZATION COMPLETE")
        print("Files modified:")
        for file_path in modified_files:
            print(f"  - {file_path}")
        
        print(f"\nBackup files created:")
        for file_path in modified_files:
            print(f"  - {file_path}.backup")
        
        print(f"\n🎯 NEXT STEPS:")
        print("1. Update config: set mini_batch_size=1 in ttt_memory_optimized.yaml")
        print("2. Run training: torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py configs/ttt_memory_optimized.yaml")
        print("3. Expected result: Memory usage reduced from 47.38GB to ~42-44GB")
    else:
        print("❌ No files were modified - torch.compile may already be disabled")

def restore_torch_compile():
    """Restore torch.compile from backup files"""
    print("🔄 RESTORING TORCH.COMPILE FROM BACKUPS")
    print("=" * 80)
    
    backup_files = list(Path(".").rglob("*.py.backup"))
    
    if not backup_files:
        print("❌ No backup files found")
        return
    
    for backup_path in backup_files:
        original_path = backup_path.with_suffix('')
        print(f"Restoring {original_path} from {backup_path}")
        
        with open(backup_path, 'r') as f:
            content = f.read()
        
        with open(original_path, 'w') as f:
            f.write(content)
    
    print(f"\n✅ Restored {len(backup_files)} files from backups")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "restore":
        restore_torch_compile()
    else:
        optimize_ttt_memory()