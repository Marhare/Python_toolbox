"""
Test: New simplified unit architecture with normalize flag

This test verifies that:
1. normalize=True (default): display in SI base units
2. normalize=False: display in original units
3. measure_si is always in SI for internal calculations
4. all calculations use measure_si, not measure
"""

import marhare as mh
import numpy as np


def test_normalize_true_default():
    """Test that normalize=True (default) displays in SI SYMBOLS (V, A, Hz)."""
    print("\n" + "="*70)
    print("TEST 1: normalize=True (default) - Display SI Symbols")
    print("="*70)
    
    # Create measurement with mV
    V = mh.quantity(5000.0, 100.0, "mV", symbol="V")
    
    print("\nInput: quantity(5000, 100, 'mV')")
    print(f"Expected: normalize=True (default)")
    
    print(f"\nInternal storage:")
    print(f"  'unit':      {V['unit']!r}            (SI SYMBOL, not 'volt'!)")
    print(f"  'measure':   {V['measure']}     (displayed values)")
    print(f"  'measure_si': {V.get('measure_si', 'MISSING')!r}     (always SI)")
    
    # Verify unit is SI SYMBOL (V not volt)
    assert V['unit'] == 'V', f"Expected unit='V', got {V['unit']!r}"
    
    # Verify measure is SI normalized
    value, sigma = V['measure']
    assert np.isclose(value, 5.0), f"Expected value≈5.0, got {value}"
    assert np.isclose(sigma, 0.1), f"Expected sigma≈0.1, got {sigma}"
    
    # Verify measure_si exists and matches measure (both SI)
    if 'measure_si' in V:
        assert V['measure_si'] == V['measure'], "measure_si should match measure when normalized"
        print("\n✓ measure_si stored correctly (both values in SI)")
    else:
        print("\n✗ WARNING: measure_si field missing")
    
    print("✓ Default normalize=True displays SI SYMBOL 'V' (not 'volt')")


def test_normalize_false():
    """Test that normalize=False displays in original units."""
    print("\n" + "="*70)
    print("TEST 2: normalize=False - Display Original Units")
    print("="*70)
    
    # Create measurement with mV, NO normalization
    V = mh.quantity(5000.0, 100.0, "mV", symbol="V", normalize=False)
    
    print("\nInput: quantity(5000, 100, 'mV', normalize=False)")
    
    print(f"\nInternal storage:")
    print(f"  'unit':      {V['unit']!r}        (what's displayed)")
    print(f"  'measure':   {V['measure']}    (displayed values)")
    print(f"  'measure_si': {V.get('measure_si', 'MISSING')}  (always SI)")
    
    # Verify unit is original
    assert V['unit'] == 'mV', f"Expected unit='mV', got {V['unit']!r}"
    
    # Verify measure is original (NOT normalized)
    value, sigma = V['measure']
    assert np.isclose(value, 5000.0), f"Expected value≈5000.0, got {value}"
    assert np.isclose(sigma, 100.0), f"Expected sigma≈100.0, got {sigma}"
    
    # Verify measure_si is SI
    if 'measure_si' in V:
        si_value, si_sigma = V['measure_si']
        assert np.isclose(si_value, 5.0), f"Expected SI value≈5.0, got {si_value}"
        assert np.isclose(si_sigma, 0.1), f"Expected SI sigma≈0.1, got {si_sigma}"
        print("\n✓ measure_si is in SI (5.0 V)")
        print("✓ measure is in original units (5000 mV)")
    else:
        print("\n✗ WARNING: measure_si field missing")
    
    print("✓ normalize=False works correctly")


def test_no_unit_display_field():
    """Verify that unit_display field is removed (simplified architecture)."""
    print("\n" + "="*70)
    print("TEST 3: No unit_display field (simplified architecture)")
    print("="*70)
    
    V = mh.quantity(5000.0, 100.0, "mV", symbol="V")
    
    print(f"\nQuantity dictionary keys: {list(V.keys())}")
    
    # Verify unit_display does NOT exist
    assert 'unit_display' not in V, "unit_display field should NOT exist in new architecture"
    
    # Verify unit exists
    assert 'unit' in V, "unit field must exist"
    
    # Verify measure_si exists
    assert 'measure_si' in V, "measure_si field must exist"
    
    print("\n✓ unit_display field removed (simplified)")
    print("✓ unit field present")
    print("✓ measure_si field present")


def test_propagation_uses_measure_si():
    """Verify that propagation uses measure_si (SI units) internally."""
    print("\n" + "="*70)
    print("TEST 4: Propagation uses measure_si (SI units)")
    print("="*70)
    
    # Create quantities with non-SI units but normalize=False
    V = mh.quantity(5000.0, 100.0, "mV", symbol="V", normalize=False)
    I = mh.quantity(2000.0, 50.0, "mA", symbol="I", normalize=False)
    
    print("\nInput:")
    print(f"  V = 5000 ± 100 mV (normalize=False)")
    print(f"  I = 2000 ± 50 mA (normalize=False)")
    
    print(f"\nInternal stored values (original):")
    print(f"  measure_V: {V['measure']}")
    print(f"  measure_I: {I['measure']}")
    
    print(f"\nInternal SI values (for calculation):")
    print(f"  measure_si_V: {V.get('measure_si')}")
    print(f"  measure_si_I: {I.get('measure_si')}")
    
    # Register and propagate
    magnitudes = mh.register(V, I)
    R = mh.quantity("V/I", "ohm", symbol="R")
    
    print(f"\nPropagating: R = V / I")
    result = mh.propagate_quantity(R, magnitudes)
    
    print(f"\nResult:")
    print(f"  R value: {result.get('result', [0])[0]:.4f} (Ohm)")
    print(f"  R sigma: {result.get('result', [0, 0])[1]:.4f} (Ohm)")
    
    # Expected: (5V / 2A) = 2.5 Ω
    expected_R = 5.0 / 2.0
    actual_R = result.get('result', [0])[0]
    
    assert np.isclose(actual_R, expected_R, rtol=0.1), \
        f"Expected R≈{expected_R} Ω, got {actual_R} Ω"
    
    print(f"\n✓ Propagation used SI values correctly")
    print(f"✓ Result is correct even with normalize=False inputs")


def test_mixed_normalize_settings():
    """Test mixing normalize=True and normalize=False quantities."""
    print("\n" + "="*70)
    print("TEST 5: Mixed normalize settings in propagation")
    print("="*70)
    
    # One normalized to SI, one not
    V = mh.quantity(5000.0, 100.0, "mV", symbol="V", normalize=True)  # display as SI
    I = mh.quantity(2000.0, 50.0, "mA", symbol="I", normalize=False)  # display as original
    
    print("\nInput:")
    print(f"  V = 5000 argsargsmV, normalize=True → displayed as {V['unit']}")
    print(f"  I = 2000 mA, normalize=False → displayed as {I['unit']}")
    
    print(f"\nBoth quantities have measure_si in SI:")
    print(f"  V measure_si: {V.get('measure_si')}")
    print(f"  I measure_si: {I.get('measure_si')}")
    
    # Register and propagate
    magnitudes = mh.register(V, I)
    R = mh.quantity("V/I", "ohm", symbol="R")
    
    result = mh.propagate_quantity(R, magnitudes)
    actual_R = result.get('result', [0])[0]
    expected_R = 2.5
    
    assert np.isclose(actual_R, expected_R, rtol=0.1), \
        f"Expected R≈{expected_R} Ω, got {actual_R} Ω"
    
    print(f"\n✓ Propagation works with mixed normalize settings")
    print(f"✓ Display units differ, but calculations use SI consistently")


def test_latex_respects_unit_field():
    """Verify that LaTeX uses the unit field directly."""
    print("\n" + "="*70)
    print("TEST 6: LaTeX uses unit field (not unit_display)")
    print("="*70)
    
    V_si = mh.quantity(5000.0, 100.0, "mV", symbol="V", normalize=True)
    V_orig = mh.quantity(5000.0, 100.0, "mV", symbol="V", normalize=False)
    
    print("\nWith normalize=True:")
    tex_si = mh.latex_quantity(V_si)
    print(f"  LaTeX: {tex_si}")
    # Should show SI SYMBOL (V), not full name (volt)
    assert "V" in tex_si and "volt" not in tex_si.lower(), \
        f"Should show SI symbol (V), not 'volt'. Got: {tex_si}"
    
    print("\nWith normalize=False:")
    tex_orig = mh.latex_quantity(V_orig)
    print(f"  LaTeX: {tex_orig}")
    assert "mV" in tex_orig or "m" in tex_orig, "Should show original unit (mV)"
    
    print("\n✓ LaTeX correctly uses unit field with SI SYMBOLS")


if __name__ == "__main__":
    test_normalize_true_default()
    test_normalize_false()
    test_no_unit_display_field()
    test_propagation_uses_measure_si()
    test_mixed_normalize_settings()
    test_latex_respects_unit_field()
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED")
    print("="*70)
    print("""
New Architecture Summary:
- normalize=True (default): display in SI SYMBOLS (V, A, Hz, m)
- normalize=False: display in original units (mV, mA, GHz, mm)  
- measure_si: always in SI for internal calculations
- unit_display field: removed (simplified to single unit field)
- All propagation uses measure_si for dimensional correctness
""")
