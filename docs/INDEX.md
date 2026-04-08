# Documentation Index

## Main Guides

- [Main README](../README.md): project overview and quick start
- [Migration Guide](MIGRATION_GUIDE.md): V1 to V2 migration and compatibility
- [Quantities Guide](README_uncertainties.md): quantities computation layer
- [LaTeX Tools Guide](README_latex_tools.md): latex presentation layer
- [Statistics Guide](README_statistics.md): statistical workflows
- [Fitting Guide](README_fitting.md): fitting workflows
- [Functions Guide](README_functions.md): math and symbolic tools
- [Graphics Guide](README_graphics.md): plotting and visualization
- [FFT Tools Guide](README_fft_tools.md): Fourier transform tools
- [Monte Carlo Guide](README_monte_carlo.md): random simulation workflows
- [Animations Guide](README_animations.md): animation tools

## Legacy Compatibility (V1)

Legacy V1 methods and imports still work through compatibility aliases.
For new code, V2 is the recommended path.

- [Migration Guide](MIGRATION_GUIDE.md)
- [V1 LaTeX Tools Changelog](CHANGELOG_V1_LATEXTOOLS.md)
- [V1 Changes Summary](SUMMARY_V1_CHANGES.md)
- [V1 Uncertainties Contract](UNCERTAINTIES_V1_CONTRACT.md)
- [V1 Uncertainties Release](UNCERTAINTIES_V1_RELEASE.md)
- [V1 Uncertainties Summary](UNCERTAINTIES_V1_SUMMARY.md)

## Technical Notes

- [Unit Conversion Implementation](UNIT_CONVERSION_IMPLEMENTATION.md)

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
