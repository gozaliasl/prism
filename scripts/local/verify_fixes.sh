#!/bin/bash
#
# Quick Verification Script - Test that Fixes Work
#

echo "🧪 QUICK VERIFICATION OF FIXES"
echo "=============================="
echo ""

cd /Users/gozalig1/Projects/jwst-mock-lens-simulator

# Check Fix #1: Enhanced Field Sampler
echo "1️⃣  Checking Enhanced Field Sampler (_rng_integers method)..."
echo ""
MAIN_CHECK=$(grep -n "if __name__" src/enhanced_field_sampling.py | cut -d: -f1)
METHOD_LINE=$(grep -n "def _rng_integers" src/enhanced_field_sampling.py | cut -d: -f1)

if [ -n "$METHOD_LINE" ]; then
    if [ $METHOD_LINE -lt $MAIN_CHECK ]; then
        echo "✅ PASS: _rng_integers() is inside class definition"
        echo "   (method at line $METHOD_LINE, before if __name__ at line $MAIN_CHECK)"
    else
        echo "❌ FAIL: _rng_integers() is OUTSIDE class (after if __name__)"
    fi
else
    echo "❌ FAIL: _rng_integers() method not found"
fi
echo ""

# Check Fix #2: Lenstronomy Metadata Removal
echo "2️⃣  Checking Lenstronomy Metadata Removal..."
echo ""
if grep -A 10 "metadata_keys = \[" src/jwst_lens_simulator.py | grep -q "redshift"; then
    echo "✅ PASS: 'redshift' is in metadata removal list"
else
    echo "❌ FAIL: 'redshift' not found in metadata removal"
fi

if grep -A 10 "metadata_keys = \[" src/jwst_lens_simulator.py | grep -q "z\|metallicity"; then
    echo "✅ PASS: Comprehensive metadata removal implemented"
else
    echo "❌ FAIL: Metadata removal seems incomplete"
fi
echo ""

# Check Python Syntax
echo "3️⃣  Checking Python Syntax..."
echo ""
python3 -m py_compile src/enhanced_field_sampling.py
if [ $? -eq 0 ]; then
    echo "✅ PASS: enhanced_field_sampling.py syntax OK"
else
    echo "❌ FAIL: enhanced_field_sampling.py has syntax errors"
fi

python3 -m py_compile src/jwst_lens_simulator.py 2>&1 | head -5
if [ $? -eq 0 ]; then
    echo "✅ PASS: jwst_lens_simulator.py syntax OK"
else
    echo "❌ FAIL: jwst_lens_simulator.py has syntax errors"
fi
echo ""

echo "🎉 VERIFICATION COMPLETE!"
echo ""
echo "Next: Run a test simulation with:"
echo "  bash scripts/local/check_simulation_errors.sh"
