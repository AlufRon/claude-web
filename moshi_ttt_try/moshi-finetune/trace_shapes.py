"""Trace TTT tensor shapes to understand the flow"""

# Before reshape: XK shape is [B, L, num_heads, head_dim]
# Example: [1, 64, 32, 128]

# After reshape_to_mini_batch (line 329-331):
# XQ = XQ.reshape(B, self.num_heads, num_mini_batch, self.mini_batch_size, self.head_dim)
# With L=64, mini_batch_size=64: num_mini_batch = 64 // 64 = 1
# New shape: [1, 32, 1, 64, 128]
#             ^B  ^heads ^num_mb ^chunk ^head_dim

# After permute (line 672):
# inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)
# Shape becomes: [1, 1, 32, 64, 128]
#                 ^num_mb ^B ^heads ^chunk ^head_dim

# So in scan loop (line 235-241 in utils.py):
# for i in range(num_items):  # num_items = 1 (first dimension)
#     x = {key: tensor[i] for key, tensor in xs.items()}
#     # x["XK"] shape: [1, 32, 64, 128]
#     #                 ^B ^heads ^chunk ^head_dim
#     carry, y = f(carry, x)  # Process this ONE mini-batch

print("With L=64, mini_batch_size=64:")
print("  num_mini_batch = 64 // 64 = 1")
print("  After permute, first dim = 1")
print("  Scan iterates 1 time")
print("  Each iteration processes: [B=1, heads=32, chunk_size=64, head_dim=128]")
print()
print("With L=2048, mini_batch_size=64:")
print("  num_mini_batch = 2048 // 64 = 32")
print("  After permute, first dim = 32")
print("  Scan iterates 32 times")
print("  Each iteration processes: [B=1, heads=32, chunk_size=64, head_dim=128]")
print()
print("Key insight: Each scan iteration processes the SAME chunk_size (64 tokens)")
print("The difference is HOW MANY times we iterate!")
