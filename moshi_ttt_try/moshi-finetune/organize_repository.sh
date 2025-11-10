#!/bin/bash
# Repository Organization Script
# This script organizes files in the moshi-finetune repository into logical directories

set -e  # Exit on error

echo "=========================================="
echo "Repository Organization Script"
echo "=========================================="
echo ""

# Create directory structure
echo "Creating directory structure..."

mkdir -p docs/guides
mkdir -p docs/analysis
mkdir -p docs/fixes
mkdir -p docs/implementation
mkdir -p analysis/librilight
mkdir -p analysis/memory
mkdir -p analysis/training
mkdir -p analysis/evaluation
mkdir -p debug/ttt
mkdir -p debug/checkpoint
mkdir -p debug/librilight
mkdir -p debug/general
mkdir -p evaluation/scripts
mkdir -p evaluation/figure5
mkdir -p slurm/training
mkdir -p slurm/inference
mkdir -p slurm/evaluation
mkdir -p tools/plotting
mkdir -p tools/data_prep
mkdir -p tools/monitoring
mkdir -p test_scripts/ttt
mkdir -p test_scripts/librilight
mkdir -p test_scripts/checkpoint
mkdir -p test_scripts/general

echo "✓ Directory structure created"
echo ""

# Move Documentation Files
echo "Organizing documentation..."

# User Guides
mv CHECKPOINT_USER_GUIDE.md docs/guides/ 2>/dev/null || true
mv QUICK_START.md docs/guides/ 2>/dev/null || true
mv PAPER_METRICS_EVALUATION_GUIDE.md docs/guides/ 2>/dev/null || true
mv PAPER_METRICS_USAGE.md docs/guides/ 2>/dev/null || true
mv QUICK_PAPER_METRICS_USAGE.md docs/guides/ 2>/dev/null || true
mv EVALUATION_GUIDE.md docs/guides/ 2>/dev/null || true
mv FIGURE5_USAGE_GUIDE.md docs/guides/ 2>/dev/null || true
mv FIGURE5_QUICK_START.md docs/guides/ 2>/dev/null || true
mv FIGURE5_QUICK_START_GUIDE.md docs/guides/ 2>/dev/null || true
mv FIGURE5_QUICK_REFERENCE.md docs/guides/ 2>/dev/null || true
mv GATING_TEST_GUIDE.md docs/guides/ 2>/dev/null || true
mv BASELINE_INFERENCE_GUIDE.md docs/guides/ 2>/dev/null || true
mv LIBRILIGHT_MIGRATION_GUIDE.md docs/guides/ 2>/dev/null || true
mv JSON_RESULTS_README.md docs/guides/ 2>/dev/null || true
mv PERSISTENCE_CHECK_README.md docs/guides/ 2>/dev/null || true

# Analysis documents
mv CHECKPOINT_ANALYSIS.md docs/analysis/ 2>/dev/null || true
mv CHECKPOINT_ANALYSIS_SUMMARY.md docs/analysis/ 2>/dev/null || true
mv BACKWARD_GATING_ANALYSIS.md docs/analysis/ 2>/dev/null || true
mv TTT_CHECKPOINT_ERROR_ANALYSIS.md docs/analysis/ 2>/dev/null || true
mv LIBRILIGHT_DATA_LOADING_ANALYSIS.md docs/analysis/ 2>/dev/null || true
mv LIBRILIGHT_IMPLEMENTATION_ANALYSIS.md docs/analysis/ 2>/dev/null || true
mv LIBRILIGHT_LOSS_ANALYSIS.md docs/analysis/ 2>/dev/null || true
mv LIBRILIGHT_METRIC_EXPLAINED.md docs/analysis/ 2>/dev/null || true
mv LIBRILIGHT_LOSS_CALCULATION_EXPLAINED.md docs/analysis/ 2>/dev/null || true
mv TTT_MEMORY_COMPLETE_ANALYSIS.md docs/analysis/ 2>/dev/null || true
mv TTT_MEMORY_OPTIMIZATION_PLAN.md docs/analysis/ 2>/dev/null || true
mv TRAINING_CODE_ANALYSIS.md docs/analysis/ 2>/dev/null || true
mv CODE_ARCHITECTURE_DEEP_DIVE.md docs/analysis/ 2>/dev/null || true
mv MOSHI_VS_VIDEODIT_ANALYSIS.md docs/analysis/ 2>/dev/null || true
mv MOSHI_STREAMING_ARCHITECTURE_UNDERSTANDING.md docs/analysis/ 2>/dev/null || true
mv CONVERGING_LINES_EXPLAINED.md docs/analysis/ 2>/dev/null || true
mv DRIFT_GROWTH_EXPLAINED.md docs/analysis/ 2>/dev/null || true
mv EXPECTED_TTT_RESULTS.md docs/analysis/ 2>/dev/null || true
mv JOB_7332661_FAILURE_ANALYSIS.md docs/analysis/ 2>/dev/null || true
mv LOG_ANALYSIS_7237224.md docs/analysis/ 2>/dev/null || true
mv INVESTIGATION_LOG_7204342.md docs/analysis/ 2>/dev/null || true
mv CHECKPOINT_300_ANALYSIS.md docs/analysis/ 2>/dev/null || true
mv CHECKPOINT_001300_COMPLETE_ANALYSIS.md docs/analysis/ 2>/dev/null || true

# Bug fixes and solutions
mv BFLOAT16_ROPE_FIX.md docs/fixes/ 2>/dev/null || true
mv CHECKPOINT_ERROR_FIX.md docs/fixes/ 2>/dev/null || true
mv CHECKPOINT_CONFIG_SOLUTION.md docs/fixes/ 2>/dev/null || true
mv DEVICE_MISMATCH_FIX.md docs/fixes/ 2>/dev/null || true
mv DTYPE_FIX_VIDEO_DIT_ANALYSIS.md docs/fixes/ 2>/dev/null || true
mv FIGURE5_BUG_FIX.md docs/fixes/ 2>/dev/null || true
mv FIGURE5_FIX_SUMMARY.md docs/fixes/ 2>/dev/null || true
mv GPU_SLURM_FIX.md docs/fixes/ 2>/dev/null || true
mv LIBRILIGHT_FIX_GUIDE.md docs/fixes/ 2>/dev/null || true
mv LIBRILIGHT_AUDIO_ONLY_FIX.md docs/fixes/ 2>/dev/null || true
mv LIBRILIGHT_AUDIO_ONLY_ISSUE.md docs/fixes/ 2>/dev/null || true
mv LIBRILIGHT_AUTOREGRESSIVE_FIX.md docs/fixes/ 2>/dev/null || true
mv LIBRILIGHT_CUDA_ERROR_FIX.md docs/fixes/ 2>/dev/null || true
mv LIBRILIGHT_FIX_COMPLETE.md docs/fixes/ 2>/dev/null || true
mv LIBRILIGHT_LOSS_BUG_FINAL_FIX.md docs/fixes/ 2>/dev/null || true
mv LIBRILIGHT_LOSS_FIX.md docs/fixes/ 2>/dev/null || true
mv LIBRILIGHT_SILENCE_BUG.md docs/fixes/ 2>/dev/null || true
mv MULTILAYER_BUG_FIX.md docs/fixes/ 2>/dev/null || true
mv MULTILAYER_FIX_COMPLETE.md docs/fixes/ 2>/dev/null || true
mv MULTILAYER_FORWARD_SIGNATURE_FIX.md docs/fixes/ 2>/dev/null || true
mv MULTILAYER_TRAINABLE_PARAMS_FIX.md docs/fixes/ 2>/dev/null || true
mv TTT_EXPLOSION_BUG_FIX.md docs/fixes/ 2>/dev/null || true
mv TTT_EXPLOSION_FIX_V2.md docs/fixes/ 2>/dev/null || true
mv TTT_EXPLOSION_ROOT_CAUSE.md docs/fixes/ 2>/dev/null || true
mv PERSISTENCE_LOGGING_FIX.md docs/fixes/ 2>/dev/null || true
mv PAPER_METRICS_FIXES_APPLIED.md docs/fixes/ 2>/dev/null || true

# Implementation documents
mv MOSHI_TTT_COMPLETE_IMPLEMENTATION_REPORT.md docs/implementation/ 2>/dev/null || true
mv TTT_MOSHI_INTEGRATION_REPORT.md docs/implementation/ 2>/dev/null || true
mv CHECKPOINT_FIX_IMPLEMENTATION_SUMMARY.md docs/implementation/ 2>/dev/null || true
mv FIGURE5_IMPLEMENTATION_COMPLETE.md docs/implementation/ 2>/dev/null || true
mv FIGURE5_IMPLEMENTATION_GUIDE.md docs/implementation/ 2>/dev/null || true
mv FIGURE5_IMPLEMENTATION_STATUS.md docs/implementation/ 2>/dev/null || true
mv FIGURE5_READY.md docs/implementation/ 2>/dev/null || true
mv INNER_LOOP_COMPLETE.md docs/implementation/ 2>/dev/null || true
mv INNER_LOOP_IMPLEMENTATION.md docs/implementation/ 2>/dev/null || true
mv INNER_LOOP_INTEGRATION_STATUS.md docs/implementation/ 2>/dev/null || true
mv LIBRILIGHT_IMPLEMENTATION_COMPLETE.md docs/implementation/ 2>/dev/null || true
mv LIBRILIGHT_TTT_SUPPORT.md docs/implementation/ 2>/dev/null || true
mv LIBRILIGHT_TTT_SOLUTION.md docs/implementation/ 2>/dev/null || true
mv LIBRILIGHT_TTT_SUCCESS.md docs/implementation/ 2>/dev/null || true
mv TTT_MEMORY_FIX_IMPLEMENTATION.md docs/implementation/ 2>/dev/null || true
mv TTT_SPIKE_OPTIMIZATION_COMPLETE.md docs/implementation/ 2>/dev/null || true
mv SILENCE_CODES_INTEGRATION.md docs/implementation/ 2>/dev/null || true
mv SILENCE_CODES_MINIMAL_CHANGES.md docs/implementation/ 2>/dev/null || true
mv LIBRILIGHT_STREAMING_FIX_PLAN.md docs/implementation/ 2>/dev/null || true
mv COMPLETE_LIBRILIGHT_PLAN.md docs/implementation/ 2>/dev/null || true
mv LIBRILIGHT_TRAINING_PLAN.md docs/implementation/ 2>/dev/null || true
mv MOSHI_FINETUNE_TTT_COMPLETE_GUIDE.md docs/implementation/ 2>/dev/null || true
mv MOSHI_TTT_COMPLETE_FLOW.md docs/implementation/ 2>/dev/null || true
mv TTT_GATING_ALPHA_LOG_IMPLEMENTATION.md docs/implementation/ 2>/dev/null || true
mv TTT_LEARNING_RATE_INSIGHTS.md docs/implementation/ 2>/dev/null || true
mv TTT_LIBRILIGHT_IMPLEMENTATION_GUIDE.md docs/implementation/ 2>/dev/null || true
mv TTT_PARAMETER_OPTIMIZATION_INVESTIGATION.md docs/implementation/ 2>/dev/null || true
mv TTT_PERSISTENCE_VERIFICATION_PLAN.md docs/implementation/ 2>/dev/null || true
mv TTT_SAVE_RESTORE_PLAN.md docs/implementation/ 2>/dev/null || true
mv TTT_STREAMING_MINIBATCH_ANALYSIS.md docs/implementation/ 2>/dev/null || true
mv TTT_NORMALIZATION_FLOW.md docs/implementation/ 2>/dev/null || true
mv TTT_NOT_TALKATIVE_SOLUTION.md docs/implementation/ 2>/dev/null || true
mv TTT_GATING_EXPERIMENT.md docs/implementation/ 2>/dev/null || true
mv LIBRILIGHT_CLEANUP_SUMMARY.md docs/implementation/ 2>/dev/null || true
mv LIBRILIGHT_STATUS.md docs/implementation/ 2>/dev/null || true
mv LIBRILIGHT_ISSUES_SUMMARY.md docs/implementation/ 2>/dev/null || true
mv LIBRILIGHT_SHUFFLE_ANSWER.md docs/implementation/ 2>/dev/null || true
mv FIGURE5_AUTO_INTEGRATION.md docs/implementation/ 2>/dev/null || true
mv FIGURE5_DIAGNOSTIC_GUIDE.md docs/implementation/ 2>/dev/null || true
mv TTT_FIGURE5_DIAGNOSTIC_SUMMARY.md docs/implementation/ 2>/dev/null || true
mv FIGURE5_BUG_FOUND.md docs/implementation/ 2>/dev/null || true
mv FIGURE5_EXECUTION_ORDER.md docs/implementation/ 2>/dev/null || true
mv FIGURE5_EVALUATION_USAGE.md docs/implementation/ 2>/dev/null || true
mv FIGURE5_RESULTS_ANALYSIS.md docs/implementation/ 2>/dev/null || true
mv MULTILAYER_LOGGING_SUMMARY.md docs/implementation/ 2>/dev/null || true
mv PAPER_METRICS_STATUS.md docs/implementation/ 2>/dev/null || true
mv CRITICAL_BUG_PLACE_INTO.md docs/implementation/ 2>/dev/null || true
mv COMPLETE_SHAPE_ISSUE_INVESTIGATION.md docs/implementation/ 2>/dev/null || true
mv SHAPE_ISSUE_DETAILED_ANALYSIS.md docs/implementation/ 2>/dev/null || true
mv TEST_CONFIG_COMPARISON.md docs/implementation/ 2>/dev/null || true
mv DASHBOARD_PLAN.md docs/implementation/ 2>/dev/null || true
mv LONG_TRAINING_INFO.md docs/implementation/ 2>/dev/null || true
mv LOSS_COMPUTATION_VERIFICATION.md docs/implementation/ 2>/dev/null || true
mv BIDIRECTIONAL_TTT_ANALYSIS.md docs/implementation/ 2>/dev/null || true
mv BIDIRECTIONAL_TTT_IMPLEMENTATION.md docs/implementation/ 2>/dev/null || true

# Small summaries
mv config_fix_summary.md docs/ 2>/dev/null || true
mv final_summary.md docs/ 2>/dev/null || true
mv librilight_results_comparison.md docs/ 2>/dev/null || true
mv log-analysis-summary.md docs/ 2>/dev/null || true
mv moshi-finetune-complete-analysis.md docs/ 2>/dev/null || true
mv librilight_problem_diagram.md docs/ 2>/dev/null || true

echo "✓ Documentation organized"

# Move Analysis Scripts
echo "Organizing analysis scripts..."

mv analyze_actual_librilight_results.py analysis/librilight/ 2>/dev/null || true
mv analyze_librilight_results.py analysis/librilight/ 2>/dev/null || true
mv analyze_memory_patterns.py analysis/memory/ 2>/dev/null || true
mv analyze_results.py analysis/evaluation/ 2>/dev/null || true
mv analyze_ttt_advantage.py analysis/training/ 2>/dev/null || true
mv analyze_ttt_parameters.py analysis/training/ 2>/dev/null || true
mv batch_parse_logs.py analysis/training/ 2>/dev/null || true
mv parse_training_log.py analysis/training/ 2>/dev/null || true
mv convert_logs_to_evaluation_format.py analysis/evaluation/ 2>/dev/null || true
mv checkpoint_summary.py analysis/evaluation/ 2>/dev/null || true

echo "✓ Analysis scripts organized"

# Move Debug Scripts
echo "Organizing debug scripts..."

# TTT Debug
mv debug_apply_ttt_step_by_step.py debug/ttt/ 2>/dev/null || true
mv debug_ttt_execution.py debug/ttt/ 2>/dev/null || true
mv debug_ttt_mlp_failure.py debug/ttt/ 2>/dev/null || true
mv debug_ttt_states.py debug/ttt/ 2>/dev/null || true
mv debug_live_ttt_execution.py debug/ttt/ 2>/dev/null || true
mv debug_gradient_flow.py debug/ttt/ 2>/dev/null || true
mv debug_gating_alpha.py debug/ttt/ 2>/dev/null || true
mv debug_optimizer_params.py debug/ttt/ 2>/dev/null || true
mv debug_inner_loop.py debug/ttt/ 2>/dev/null || true
mv debug_losses.py debug/ttt/ 2>/dev/null || true
mv debug_ln_explosion.py debug/ttt/ 2>/dev/null || true

# Checkpoint Debug
mv debug_checkpoint_fix_verification.py debug/checkpoint/ 2>/dev/null || true
mv check_checkpoint_dtypes.py debug/checkpoint/ 2>/dev/null || true
mv check_model_dtypes.py debug/checkpoint/ 2>/dev/null || true
mv check_ttt_training.py debug/checkpoint/ 2>/dev/null || true

# LibriLight Debug
mv debug_librilight_tokens.py debug/librilight/ 2>/dev/null || true
mv debug_token_ids.py debug/librilight/ 2>/dev/null || true

# General Debug
mv debug_production_training.py debug/general/ 2>/dev/null || true
mv debug_dtype_mismatch.py debug/general/ 2>/dev/null || true
mv find_dtype_issue.py debug/general/ 2>/dev/null || true
mv simple_dtype_debug.py debug/general/ 2>/dev/null || true

echo "✓ Debug scripts organized"

# Move Evaluation Scripts
echo "Organizing evaluation scripts..."

mv eval_figure5_batch.py evaluation/figure5/ 2>/dev/null || true
mv eval_from_checkpoint.py evaluation/scripts/ 2>/dev/null || true
mv evaluate_checkpoint_figure5.py evaluation/figure5/ 2>/dev/null || true
mv evaluate_checkpoint_with_inner_loop.py evaluation/scripts/ 2>/dev/null || true
mv run_eval_quick.py evaluation/scripts/ 2>/dev/null || true
mv run_eval_with_inner_loop.sh evaluation/scripts/ 2>/dev/null || true
mv run_paper_metrics_on_checkpoint.py evaluation/scripts/ 2>/dev/null || true
mv validate_paper_metrics.py evaluation/scripts/ 2>/dev/null || true
mv verify_state_persistence.py evaluation/scripts/ 2>/dev/null || true
mv verify_training_code_version.py evaluation/scripts/ 2>/dev/null || true
mv verify_ttt_integration_status.py evaluation/scripts/ 2>/dev/null || true

echo "✓ Evaluation scripts organized"

# Move SLURM Scripts
echo "Organizing SLURM scripts..."

# Training
mv train_moshi_ttt.slurm slurm/training/ 2>/dev/null || true
mv test_gating_alpha.slurm slurm/training/ 2>/dev/null || true

# Inference
mv run_baseline_inference.slurm slurm/inference/ 2>/dev/null || true
mv run_inference_ttt.slurm slurm/inference/ 2>/dev/null || true
mv run_inference_ttt_talkative.slurm slurm/inference/ 2>/dev/null || true
mv submit_baseline_inference.sh slurm/inference/ 2>/dev/null || true
mv submit_inference.sh slurm/inference/ 2>/dev/null || true
mv submit_inference_balanced.sh slurm/inference/ 2>/dev/null || true
mv submit_inference_talkative.sh slurm/inference/ 2>/dev/null || true
mv submit_gating_test.sh slurm/inference/ 2>/dev/null || true
mv submit_job.sh slurm/inference/ 2>/dev/null || true

# Evaluation
mv run_paper_metrics.slurm slurm/evaluation/ 2>/dev/null || true
mv submit_paper_metrics.sh slurm/evaluation/ 2>/dev/null || true
mv submit_paper_metrics_wrapper.sh slurm/evaluation/ 2>/dev/null || true
mv run_figure5_diagnostic.slurm slurm/evaluation/ 2>/dev/null || true
mv submit_figure5_diagnostic.sh slurm/evaluation/ 2>/dev/null || true

echo "✓ SLURM scripts organized"

# Move Tool Scripts
echo "Organizing tool scripts..."

# Plotting
mv plot_all_training_runs.py tools/plotting/ 2>/dev/null || true
mv plot_enhanced_comparison.py tools/plotting/ 2>/dev/null || true
mv plot_evaluation_results.py tools/plotting/ 2>/dev/null || true
mv plot_librilight_results.py tools/plotting/ 2>/dev/null || true
mv plot_robust_evaluation_results.py tools/plotting/ 2>/dev/null || true
mv plot_ttt_advantage_summary.py tools/plotting/ 2>/dev/null || true
mv create_simple_ttt_plot.py tools/plotting/ 2>/dev/null || true
mv create_interactive_table.py tools/plotting/ 2>/dev/null || true
mv create_results_table.py tools/plotting/ 2>/dev/null || true
mv experimental_comparison.py tools/plotting/ 2>/dev/null || true
mv extract_2000_sample_results.py tools/plotting/ 2>/dev/null || true

# Data Prep
mv calculate_fair_lora_rank.py tools/data_prep/ 2>/dev/null || true
mv annotate.py tools/data_prep/ 2>/dev/null || true
mv create_1hour_librilight.py tools/data_prep/ 2>/dev/null || true

# Monitoring
mv monitor_long_training.sh tools/monitoring/ 2>/dev/null || true
mv final_fixed_ttt_parameter_tracker.py tools/monitoring/ 2>/dev/null || true
mv fixed_ttt_parameter_tracker.py tools/monitoring/ 2>/dev/null || true

echo "✓ Tool scripts organized"

# Move Test Scripts
echo "Organizing test scripts..."

# TTT Tests
mv test_ttt_audio_rope.py test_scripts/ttt/ 2>/dev/null || true
mv test_ttt_cuda_graph_fix.py test_scripts/ttt/ 2>/dev/null || true
mv test_ttt_data_leakage_fix.py test_scripts/ttt/ 2>/dev/null || true
mv test_ttt_gating.sh test_scripts/ttt/ 2>/dev/null || true
mv test_ttt_norm_fix.py test_scripts/ttt/ 2>/dev/null || true
mv test_ttt_parameters_update.py test_scripts/ttt/ 2>/dev/null || true
mv test_ttt_persistence_bug.py test_scripts/ttt/ 2>/dev/null || true
mv test_ttt_projections.py test_scripts/ttt/ 2>/dev/null || true
mv test_ttt_reconstruction_bug.py test_scripts/ttt/ 2>/dev/null || true
mv test_ttt_reset_investigation.py test_scripts/ttt/ 2>/dev/null || true
mv test_ttt_states_actually_update.py test_scripts/ttt/ 2>/dev/null || true
mv test_multi_layer_ttt.py test_scripts/ttt/ 2>/dev/null || true

# LibriLight Tests
mv test_librilight_evaluation.py test_scripts/librilight/ 2>/dev/null || true
mv test_librilight_fix.py test_scripts/librilight/ 2>/dev/null || true
mv test_librilight_integration.py test_scripts/librilight/ 2>/dev/null || true
mv test_librilight_logits_fix.py test_scripts/librilight/ 2>/dev/null || true
mv test_fixed_librilight.py test_scripts/librilight/ 2>/dev/null || true
mv test_frozen_moshi_librilight.py test_scripts/librilight/ 2>/dev/null || true
mv test_real_librilight_evaluation.py test_scripts/librilight/ 2>/dev/null || true
mv test_quick_librilight_fix.py test_scripts/librilight/ 2>/dev/null || true

# Checkpoint Tests
mv test_checkpoint_fix.py test_scripts/checkpoint/ 2>/dev/null || true
mv test_checkpoint_verification.py test_scripts/checkpoint/ 2>/dev/null || true
mv test_enhanced_checkpoint_system.py test_scripts/checkpoint/ 2>/dev/null || true
mv test_save_restore_fix.py test_scripts/checkpoint/ 2>/dev/null || true
mv test_simple_save_restore.py test_scripts/checkpoint/ 2>/dev/null || true

# General Tests
mv test_attention_patterns.py test_scripts/general/ 2>/dev/null || true
mv test_backward_graph_error.py test_scripts/general/ 2>/dev/null || true
mv test_debug_logging_fix.py test_scripts/general/ 2>/dev/null || true
mv test_debug_verification.py test_scripts/general/ 2>/dev/null || true
mv test_depformer_fix.py test_scripts/general/ 2>/dev/null || true
mv test_figure5_logging.py test_scripts/general/ 2>/dev/null || true
mv test_freqs_cis_fix.py test_scripts/general/ 2>/dev/null || true
mv test_frozen_moshi_baseline.py test_scripts/general/ 2>/dev/null || true
mv test_fsdp_reset_simple.py test_scripts/general/ 2>/dev/null || true
mv test_gpu_simple.py test_scripts/general/ 2>/dev/null || true
mv test_hybrid_layer_truncation.py test_scripts/general/ 2>/dev/null || true
mv test_import_correct.py test_scripts/general/ 2>/dev/null || true
mv test_import_path.py test_scripts/general/ 2>/dev/null || true
mv test_inner_loop_logging.py test_scripts/general/ 2>/dev/null || true
mv test_length_restoration.py test_scripts/general/ 2>/dev/null || true
mv test_live_logging.py test_scripts/general/ 2>/dev/null || true
mv test_loss_computation_only.py test_scripts/general/ 2>/dev/null || true
mv test_memory_optimizations.py test_scripts/general/ 2>/dev/null || true
mv test_model_reset_integration.py test_scripts/general/ 2>/dev/null || true
mv test_moshi_native_streaming.py test_scripts/general/ 2>/dev/null || true
mv test_persistence_logging.py test_scripts/general/ 2>/dev/null || true
mv test_quick_reset_validation.py test_scripts/general/ 2>/dev/null || true
mv test_silence_codes_integration.py test_scripts/general/ 2>/dev/null || true
mv test_simple_backward_error.py test_scripts/general/ 2>/dev/null || true
mv test_simple_logging.py test_scripts/general/ 2>/dev/null || true
mv test_streaming_fix.py test_scripts/general/ 2>/dev/null || true
mv test_streaming_state_replication.py test_scripts/general/ 2>/dev/null || true
mv test_training_end_evaluation.py test_scripts/general/ 2>/dev/null || true
mv test_truncation_fix.py test_scripts/general/ 2>/dev/null || true
mv test_5layer_mlp_fix.py test_scripts/general/ 2>/dev/null || true

echo "✓ Test scripts organized"

# Move Investigation/Research Scripts
echo "Organizing investigation scripts..."
mkdir -p investigation

mv investigate_depformer.py investigation/ 2>/dev/null || true
mv investigate_moshi_architecture.py investigation/ 2>/dev/null || true
mv investigate_rope_frequencies.py investigation/ 2>/dev/null || true
mv investigate_shape_issue.py investigation/ 2>/dev/null || true
mv investigate_ttt_oom.py investigation/ 2>/dev/null || true
mv determine_dimensions.py investigation/ 2>/dev/null || true
mv get_layer_dimensions.py investigation/ 2>/dev/null || true
mv get_specific_runs.py investigation/ 2>/dev/null || true
mv get_wandb_metrics.py investigation/ 2>/dev/null || true
mv audio_rope_implementation.py investigation/ 2>/dev/null || true
mv measure_ttt_rope_impact.py investigation/ 2>/dev/null || true
mv comprehensive_ttt_execution_test.py investigation/ 2>/dev/null || true
mv proposed_ttt_memory_solutions.py investigation/ 2>/dev/null || true
mv disable_torch_compile.py investigation/ 2>/dev/null || true

echo "✓ Investigation scripts organized"

# Move Training Scripts
echo "Organizing training scripts..."
mkdir -p training

mv train_ttt.py training/ 2>/dev/null || true
mv train_ttt_production.py training/ 2>/dev/null || true
mv train_ttt_single_gpu.py training/ 2>/dev/null || true
mv run_train.sh training/ 2>/dev/null || true

echo "✓ Training scripts organized"

# Move Inference Scripts
echo "Organizing inference scripts..."
mkdir -p inference

mv run_inference_with_figure5.py inference/ 2>/dev/null || true
mv run_inference_with_lora.py inference/ 2>/dev/null || true
mv run_inference_with_ttt.py inference/ 2>/dev/null || true
mv run_librilight_only.py inference/ 2>/dev/null || true
mv run_lora_inference.sh inference/ 2>/dev/null || true
mv run_ttt_inference.sh inference/ 2>/dev/null || true
mv run_ttt_minimal.sh inference/ 2>/dev/null || true
mv run_inference_no_ttt.sh inference/ 2>/dev/null || true

echo "✓ Inference scripts organized"

# Move utility scripts
echo "Organizing utility scripts..."
mkdir -p utils

mv clear_cache_and_run.sh utils/ 2>/dev/null || true

echo "✓ Utility scripts organized"

# Move JSON data files
echo "Organizing data files..."
mkdir -p data

mv [0-9].json data/ 2>/dev/null || true
mv baseline_paper_metrics_results.json data/ 2>/dev/null || true
mv ttt_analysis_report.json data/ 2>/dev/null || true
mv dailytalk.jsonl data/ 2>/dev/null || true
mv frozen_moshi_baseline.npz data/ 2>/dev/null || true

echo "✓ Data files organized"

# Move config files
echo "Organizing config files..."

mv figure5_quick.yaml example/ 2>/dev/null || true
mv optimized_ttt_configs.yaml configs/ 2>/dev/null || true
mv silence_codes_config_example.yaml configs/ 2>/dev/null || true

echo "✓ Config files organized"

echo ""
echo "=========================================="
echo "Organization Complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  docs/           - All documentation organized by type"
echo "  analysis/       - Analysis scripts for librilight, memory, training, evaluation"
echo "  debug/          - Debug scripts for ttt, checkpoint, librilight, general"
echo "  evaluation/     - Evaluation scripts and figure5 tools"
echo "  slurm/          - SLURM scripts for training, inference, evaluation"
echo "  tools/          - Plotting, data prep, and monitoring tools"
echo "  test_scripts/   - Test scripts organized by category"
echo "  investigation/  - Research and investigation scripts"
echo "  training/       - Training scripts"
echo "  inference/      - Inference scripts"
echo "  utils/          - Utility scripts"
echo "  data/           - Data and config files"
echo ""
echo "Root directory now contains only:"
echo "  - Core files (train.py, README.md, etc.)"
echo "  - Package directories (finetune/, moshi_ttt/, etc.)"
echo "  - Configuration directories (example/, configs/)"
echo ""
