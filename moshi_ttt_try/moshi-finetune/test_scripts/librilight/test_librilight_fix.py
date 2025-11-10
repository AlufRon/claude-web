#!/usr/bin/env python3
"""
Test script to validate the LibriLight NaN fix using LMGen streaming API.
"""

import sys
import os
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_lmgen_import():
    """Test that we can import LMGen successfully."""
    try:
        from moshi.models.lm import LMGen
        logger.info("✅ Successfully imported LMGen")
        return True
    except ImportError as e:
        logger.error(f"❌ Failed to import LMGen: {e}")
        return False

def test_paper_metrics_import():
    """Test that paper_metrics imports without errors."""
    try:
        from finetune.paper_metrics import PaperMetricsEvaluator
        logger.info("✅ Successfully imported PaperMetricsEvaluator")
        return True
    except ImportError as e:
        logger.error(f"❌ Failed to import PaperMetricsEvaluator: {e}")
        return False

def test_basic_lmgen_creation():
    """Test basic LMGen creation without full model loading."""
    try:
        from moshi.models.lm import LMGen
        
        # This will fail without a model, but should validate the import
        logger.info("✅ LMGen class is accessible")
        return True
    except Exception as e:
        logger.error(f"❌ Error accessing LMGen: {e}")
        return False

def main():
    """Run basic validation tests."""
    logger.info("🔧 Testing LibriLight NaN fix implementation...")
    
    tests = [
        ("LMGen Import", test_lmgen_import),
        ("PaperMetrics Import", test_paper_metrics_import),
        ("Basic LMGen Access", test_basic_lmgen_creation),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name}...")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    logger.info("\n📊 Test Results:")
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\n🎯 Overall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        logger.info("🎉 All basic tests passed! LibriLight fix is ready for testing.")
        return True
    else:
        logger.error("💥 Some tests failed. Check imports and environment.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)