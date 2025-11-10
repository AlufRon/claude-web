## 🚨 Critical Fixes Applied to paper_metrics.py

### Problems Found & Fixed:

1. **CRITICAL: Wrong Loss vs Likelihood Logic**
   - Your `_compute_likelihood()` returns LOSS (NLL) where higher = worse
   - But comparisons were treating it as likelihood where higher = better
   - **Fixed**: All comparisons now use `nll1 < nll2` (lower loss = better)

2. **CRITICAL: Incorrect Silence Codes**
   - Was using `(B, 1, samples)` instead of `(B, mimi.channels, samples)`
   - **Fixed**: Now uses correct MIMI channel count

3. **CRITICAL: Incomplete sWUGGY Pair Creation**
   - Missing voice matching logic from eval_paper
   - **Fixed**: Now uses exact eval_paper methodology with voice pairing

4. **Import Issues**
   - Added proper fallbacks for torchaudio/librosa imports
   - **Fixed**: Graceful handling when libraries not available

### Expected Results After Fix:

**Before Fix (Your Results):**
- sBLIMP: ~10-20% (should be ~55%)
- sWUGGY: ~10-30% (should be ~74%)
- Story Cloze: ~30-40% (should be ~60-70%)

**After Fix (Expected):**
- sBLIMP: ~50-60% (closer to eval_paper results)
- sWUGGY: ~70-80% (closer to eval_paper results)
- Story Cloze: ~60-75% (closer to eval_paper results)

### Key Changes:
1. `word_nll < nonword_nll` instead of `word_ll > nonword_ll`
2. `correct_nll < incorrect_nll` instead of `correct_ll > incorrect_ll`
3. Proper MIMI channel count in silence generation
4. Complete sWUGGY pair creation with voice matching

These fixes should bring your evaluation results much closer to the eval_paper benchmarks!
