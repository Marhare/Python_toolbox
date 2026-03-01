# Documentation Index

**Complete guide to Python Toolbox (marhare) documentation**

---

## 📖 User Guides (Module Documentation)

### Core Modules

| Module | Documentation | Description |
|--------|---------------|-------------|
| **uncertainties** | [README_uncertainties.md](README_uncertainties.md) | Quantities with uncertainty propagation (v1.0) |
| **statistics** | [README_statistics.md](README_statistics.md) | Statistical analysis with uncertainties |
| **monte_carlo** | [README_monte_carlo.md](README_monte_carlo.md) | Monte Carlo simulations |
| **fitting** | [README_fitting.md](README_fitting.md) | Curve fitting and regression |
| **graphics** | [README_graphics.md](README_graphics.md) | Scientific visualization |
| **latex_tools** | [README_latex_tools.md](README_latex_tools.md) | LaTeX formatting and tables |
| **fft_tools** | [README_fft_tools.md](README_fft_tools.md) | FFT and signal processing |
| **animations** | [README_animations.md](README_animations.md) | Animated plots |
| **functions** | [README_functions.md](README_functions.md) | Mathematical functions |

---

## 🏗️ Architecture Documentation

### uncertainties v1.0

| Document | Lines | Purpose |
|----------|-------|---------|
| **[README_V1.md](README_V1.md)** | ~150 | Quick start guide for v1.0 |
| **[UNCERTAINTIES_V1_RELEASE.md](UNCERTAINTIES_V1_RELEASE.md)** | 450+ | Complete release notes and changelog |
| **[UNCERTAINTIES_V1_CONTRACT.md](UNCERTAINTIES_V1_CONTRACT.md)** | 600+ | Formal architecture contract with test verification |
| **[UNCERTAINTIES_V1_SUMMARY.md](UNCERTAINTIES_V1_SUMMARY.md)** | ~300 | Implementation summary and deliverables |
| **[UNIT_CONVERSION_IMPLEMENTATION.md](UNIT_CONVERSION_IMPLEMENTATION.md)** | ~400 | Unit conversion system details |

---

## 🎯 Quick Navigation

### For New Users
1. Start with the main **[README.md](../README.md)** in repository root
2. Read **[README_uncertainties.md](README_uncertainties.md)** for the core module
3. Check **[README_V1.md](README_V1.md)** for v1.0 features

### For Developers
1. Read **[UNCERTAINTIES_V1_CONTRACT.md](UNCERTAINTIES_V1_CONTRACT.md)** for formal guarantees
2. Review **[UNCERTAINTIES_V1_SUMMARY.md](UNCERTAINTIES_V1_SUMMARY.md)** for implementation details
3. See **[UNIT_CONVERSION_IMPLEMENTATION.md](UNIT_CONVERSION_IMPLEMENTATION.md)** for unit system

### For Contributors
1. Read **[UNCERTAINTIES_V1_RELEASE.md](UNCERTAINTIES_V1_RELEASE.md)** for design decisions
2. Check test files in `../tests/` for examples
3. Follow coding patterns from existing modules

---

## 📊 Documentation Coverage

| Category | Files | Status |
|----------|-------|--------|
| User Guides | 9 | ✅ Complete |
| Architecture | 5 | ✅ Complete |
| Test Coverage | 67/67 | ✅ 100% passing |
| README Examples | 34/34 | ✅ All verified |

---

## 🔍 Document Relationships

```
README.md (root)
    │
    ├── docs/README_uncertainties.md (user guide)
    │       │
    │       ├── docs/README_V1.md (quick start)
    │       ├── docs/UNCERTAINTIES_V1_RELEASE.md (changelog)
    │       ├── docs/UNCERTAINTIES_V1_CONTRACT.md (formal spec)
    │       ├── docs/UNCERTAINTIES_V1_SUMMARY.md (implementation)
    │       └── docs/UNIT_CONVERSION_IMPLEMENTATION.md (system details)
    │
    ├── docs/README_statistics.md
    ├── docs/README_monte_carlo.md
    ├── docs/README_fitting.md
    ├── docs/README_graphics.md
    ├── docs/README_latex_tools.md
    ├── docs/README_fft_tools.md
    ├── docs/README_animations.md
    └── docs/README_functions.md
```

---

## 📝 Documentation Standards

### File Naming Convention
- `README_<module>.md` — User-facing module documentation
- `<MODULE>_V<X>_<TYPE>.md` — Version-specific architecture docs
- `<TOPIC>_IMPLEMENTATION.md` — Technical implementation details

### Documentation Types
1. **User Guides** (`README_*.md`) — Tutorial-style with examples
2. **Architecture** (`*_CONTRACT.md`) — Formal specifications
3. **Release Notes** (`*_RELEASE.md`) — Changelog and migration guides
4. **Implementation** (`*_IMPLEMENTATION.md`) — Technical deep dives

---

## 🧪 Verified Examples

All code examples in documentation are automatically tested:

```bash
# Run documentation tests
python tests/test_readme_examples.py

# Output:
# ✅ ALL README EXAMPLES WORK CORRECTLY ✅
# PASSED: 34/34
```

---

## 🗂️ File Organization

```
docs/
├── INDEX.md                              # This file
│
├── README_uncertainties.md              # User guide
├── README_statistics.md
├── README_monte_carlo.md
├── README_fitting.md
├── README_graphics.md
├── README_latex_tools.md
├── README_fft_tools.md
├── README_animations.md
├── README_functions.md
│
├── README_V1.md                         # v1.0 quick start
├── UNCERTAINTIES_V1_RELEASE.md          # v1.0 release notes
├── UNCERTAINTIES_V1_CONTRACT.md         # v1.0 formal contract
├── UNCERTAINTIES_V1_SUMMARY.md          # v1.0 implementation
├── UNIT_CONVERSION_IMPLEMENTATION.md    # Unit system details
│
└── img/                                 # Documentation images
```

---

## 📅 Last Updated

**Date:** March 1, 2026  
**Version:** 1.0  
**Status:** All documentation complete and verified ✅

---

**Questions?** Open an issue or see the main [README.md](../README.md)
