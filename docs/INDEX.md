# Documentation Index

## Main Guides

- README.md: project overview and quick start
- MIGRATION_GUIDE.md: V1 to V2 migration and compatibility
- README_uncertainties.md: quantities computation layer
- README_latex_tools.md: latex presentation layer
- README_statistics.md: statistical workflows
- README_fitting.md: fitting workflows
- README_functions.md: math/symbolic tools

## Technical Notes

- UNIT_CONVERSION_IMPLEMENTATION.md

## Import Convention (current)

Use these imports for new code:

```python
from marhare.quantities import quantity, value_quantity
from marhare.latex import valor_pm, latex_quantity, tabla_latex, exportar
```

Legacy imports still work but are deprecated:

- marhare.uncertainties
- marhare.quantities2
- marhare.propagation
- marhare.units
- marhare.latex_tools
