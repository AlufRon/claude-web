# Test Config vs Full Training Config Comparison

## Summary

✅ **Test config now has all essential features from full training config**

## Key Differences (Test vs Full)

### Training Duration
- **Test**: 5 steps
- **Full**: 200 steps

### Paper Metrics Sample Sizes
- **Test**: 10 samples each (sBLIMP, sWUGGY, tStory, sStory)
- **Full**: 5000 samples each

### LibriLight Files
- **Test**: 1 file (~1 hour audio)
- **Full**: 3 files (~3 hours audio)

### Evaluation Frequency
- **Test**: Every 5 steps (paper_metrics_freq: 5)
- **Full**: Every 200 steps (paper_metrics_freq: 200)

### Checkpointing
- **Test**: Disabled (do_ckpt: false)
- **Full**: Enabled (do_ckpt: true, ckpt_freq: 100)

### WandB
- **Test**: Offline mode
- **Full**: Online mode

### Run Directory
- **Test**: `/sise/eliyanac-group/ron_al/test_checkpoint_fix_with_fig5`
- **Full**: `/sise/eliyanac-group/ron_al/seamless_mosiworking2005`

## Features Present in Both

✅ **TTT Configuration**
- `enable: true`
- `layers: "29,30,31"`
- `base_lr: 0.5`
- `mini_batch_size: 1`
- `persistent_states: true`
- `initial_gating_alpha: 0.1`
- `ttt_mlp_layers: 2`
- `ttt_mlp_expansion_factor: 4.0`

✅ **Paper Metrics**
- `paper_metrics_eval: true`
- `save_results_json: true`
- `results_dir: "./evaluation_results"`
- `paper_metrics_use_silence: true`
- `paper_metrics_use_user_stream: false`

✅ **Figure 5 (TTT Loss Trajectories)**
- `ttt_fig5_enable: true`
- `ttt_fig5_max_T: 2048`
- `ttt_fig5_layers: [29, 30, 31]`
- `ttt_fig5_smooth: 10`

✅ **LibriLight Long Context Evaluation**
- `librilight_evaluation_mode: pre_concatenated`
- `librilight_concatenated_dir: /home/alufr/ttt_tests/librilight_1hour_sequences`

✅ **Benchmark Datasets**
- sBLIMP (syntactic evaluation)
- sWUGGY (lexical evaluation)
- tStory (temporal story evaluation)
- sStory (spatial story evaluation)

✅ **Model Configuration**
- `full_finetuning: false`
- `gradient_checkpointing: false`
- `first_codebook_weight_multiplier: 100.`
- `text_padding_weight: 0.5`

✅ **LoRA Configuration**
- `enable: false` (using TTT only)
- `rank: 128`
- `scaling: 2.`

## Test Config Purpose

The test config is designed to:
1. **Verify checkpoint fix** works (scan_checkpoint_group_size=16)
2. **Test Figure 5 generation** with minimal LibriLight data
3. **Fast execution** (~2 minutes instead of hours)
4. **All features enabled** to catch any integration issues

## Expected Test Results

1. ✅ No checkpoint errors (verified in previous test)
2. ✅ Paper metrics run successfully (verified in previous test)
3. 🔄 **Figure 5 plots generated** (testing now with LibriLight)
4. ✅ LibriLight evaluation completes
5. ✅ JSON results saved to evaluation_results/

## Files to Check After Test

- `/sise/eliyanac-group/ron_al/test_checkpoint_fix_with_fig5/evaluation_results/`
- Figure 5 plots (should be in evaluation_results or logged to WandB)
- `results.json` with all metrics
- `summary.txt` with evaluation summary

## Update: LibriLight Token Limit Added

**New Parameter**: `librilight_max_tokens: 3000`

This limits LibriLight evaluation to 3000 tokens (~24 seconds of audio) instead of the full 43,060 tokens (~5.7 minutes).

**Time Savings**:
- Full file: ~20-30 minutes
- Limited to 3000: ~2-3 minutes

**Why This is Sufficient for Testing**:
- Figure 5 tracks positions 0-2048 (ttt_fig5_max_T: 2048)
- 3000 tokens covers the entire Figure 5 range plus buffer
- Still tests the full pipeline end-to-end

**Current Run (Job 7237208)**:
- Processing full 43,060 tokens (no limit)
- Will complete eventually (~20 min total)
- Already collected Figure 5 data for first 2000+ positions

**Next Test Run**:
- Will use 3000 token limit
- Much faster (~2-3 minutes for LibriLight)
- Still generates complete Figure 5 plots
