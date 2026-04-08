import pint
ureg = pint.UnitRegistry()

# Test simplification
q1 = 1.0 * ureg.meter
q2 = 1.0 * ureg.meter / ureg.second
q3 = 1.0 * ureg.volt / ureg.ampere  # Should be ohm

print('Test unit simplification:')
print(f'q1.units: {q1.units}')
print(f'q2.units: {q2.units}')
print(f'q3.units: {q3.units}')
print(f'q3.to_compact(): {q3.to_compact().units}')

# Check dimensionality
print(f'\nCheck if units are equivalent:')
print(f'V/A simplified: {(1*ureg.volt / ureg.ampere).to_compact()}')
print(f'V/A in ohm: {(1*ureg.volt / ureg.ampere).to("ohm")}')

# Get the actual unit name
q_ohm = 1.0 * ureg.ohm
print(f'\nOhm comparison:')
print(f'Direct ohm: {q_ohm.units}')
print(f'V/A dimensionality: {q3.dimensionality}')
print(f'Ohm dimensionality: {q_ohm.dimensionality}')
print(f'Are equivalent? {q3.dimensionality == q_ohm.dimensionality}')

# Check if we can find the unit name from dimensionality
print(f'\nFind unit names from dimensionality:')
dim = q3.dimensionality
print(f'Dimension {dim}')

# Check all defined units
from pint.definitions import Definition
for name in ['ohm', 'watt', 'joule', 'pascal', 'newton', 'hertz']:
    try:
        q = 1.0 * getattr(ureg, name)
        print(f'{name}: {q.dimensionality}')
    except:
        pass
