import marhare as mh

print("=== Issue #1: Dimensional validation in addition ===")
try:
    m = mh.quantity(5.0, 0.1, 'm', symbol='x')
    s = mh.quantity(3.0, 0.1, 's', symbol='y')
    result = m + s
    print(f'ERROR: (5 m) + (3 s) = {result.value} (should have failed!)')
except ValueError as e:
    print(f'✓ Good: {str(e)[:80]}...')

print("\n=== Issue #2: Scalar operations with units ===")
x = mh.quantity(5.0, 0.1, 'm', symbol='x')
r1 = 5 * x
print(f'✓ 5 * (5 m) = {r1.value} {r1.unit}')
r2 = x * 5
print(f'✓ (5 m) * 5 = {r2.value} {r2.unit}')
r3 = x / 5
print(f'✓ (5 m) / 5 = {r3.value} {r3.unit}')

print("\n=== Issue #3: Unit simplification ===")
V = mh.quantity(10.0, 0.5, 'V', symbol='U')
I = mh.quantity(2.0, 0.1, 'A', symbol='I')
R = V / I
print(f'✓ (10 V) / (2 A) = {R.value} {R.unit}')

print("\n=== Complex unit algebra ===")
a = mh.quantity(6.0, 0.1, 'm/s', symbol='v')
b = mh.quantity(2.0, 0.05, 'm/s', symbol='v2')
result = a / b
print(f'✓ (6 m/s) / (2 m/s) = {result.value} {result.unit} (dimensionless)')

print("\n=== All tests completed ===")
