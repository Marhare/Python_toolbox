# Summary of v1.0 Changes

**Date:** March 2026  
**Impact:** LaTeX formatting and group propagation improvements  
**Backward compatible:** ✅ YES

---

## What Changed

### 1. **Tablas con Grupos como Columnas** (Groups as Table Columns)

**Localización:** [README_latex_tools.md](README_latex_tools.md#automatic-group-based-tables)

Cuando una magnitud tiene **grupos experimentales**, `latex_quantity()` ahora genera automáticamente una tabla limpia con:
- **Columnas:** Nombres de los grupos (ordenados alfabéticamente)
- **Filas:** Valores en cada grupo
- **Encabezado de fila:** Símbolo de la magnitud

```python
wavelength = mh.quantity(
    groups={"red": (...), "blue": (...), "green": (...)},
    unit="nm", symbol="λ"
)
print(mh.latex_quantity(wavelength))  # Automatic table!
```

---

### 2. **Herencia Automática de Grupos en Propagación** (Automatic Group Inheritance)

**Localización:** [README_uncertainties.md](README_uncertainties.md#group-aware-propagation-modes) y [CHANGELOG_V1_LATEXTOOLS.md](CHANGELOG_V1_LATEXTOOLS.md#2-group-inheritance-in-propagation)

**Ahora funciona:** Cuando propagas una cantidad derivada, **hereda automáticamente los grupos** incluso si no todas las magnitudes de entrada los tienen.

**Antes (v0.x):** ❌ Error si algunas magnitudes no tienen grupos  
**Ahora (v1.0+):** ✅ Magnitudes sin grupos se tratan como valores globales (se replican para todos)

**Ejemplo:**
```python
# delta_m CON grupos
delta_m = mh.quantity(groups={"rojo": (...), "amarillo": (...)}, ...)

# alpha SIN grupos (escalar)
alpha = mh.quantity(0.5, 0.1, "rad", symbol="alpha")

# Crear derivada
n = mh.quantity("sin((delta_m + alpha)/2)/sin(alpha/2)", unit="1", symbol="n")

# Propagar
registry = mh.register(delta_m, alpha, n)
n_result = mh.propagate_quantity(n, registry)

# ✅ n_result HEREDA los grupos de delta_m automáticamente!
print(n_result.groups)  # ['rojo', 'amarillo']
print(mh.latex_quantity(n_result))  # Tabla con grupos como columnas
```

---

### 3. **Tablas sin Paréntesis** (Cleaner Table Format - No Parentheses)

**Localización:** [CHANGELOG_V1_LATEXTOOLS.md](CHANGELOG_V1_LATEXTOOLS.md#31-no-parentheses-in-magnitude-tables)

**Antes (v0.x):**
```latex
m & $(1.25 \pm 0.01)\,\mathrm{kg}$ \\   ← paréntesis
```

**Ahora (v1.0+):**
```latex
m & $1.25 \pm 0.01\,\mathrm{kg}$ \\     ← sin paréntesis  
```

**Beneficio:** Menos desorden visual en tablas de publicación.

---

### 4. **Sin "(1)" para Unidades Adimensionales** (No "(1)" for Dimensionless Units)

**Localización:** [CHANGELOG_V1_LATEXTOOLS.md](CHANGELOG_V1_LATEXTOOLS.md#32-dimensionless-units-no-longer-show-1)

**Antes (v0.x):**
```python
n = mh.quantity(1.5, 0.1, unit="1", symbol="n")
print(mh.latex_quantity(n))
# Output: $n = 1.5 \pm 0.1 \, (1)$  ← awkward (1)
```

**Ahora (v1.0+):**
```python
n = mh.quantity(1.5, 0.1, unit="1", symbol="n")
print(mh.latex_quantity(n))
# Output: $n = 1.5 \pm 0.1$  ← limpio, sin unidad
```

**Beneficio:** Salida más limpia para tesis y revistas.

---

### 5. **Símbolo por Defecto "_result"** (Auto-Default Symbol)

**Localización:** [CHANGELOG_V1_LATEXTOOLS.md](CHANGELOG_V1_LATEXTOOLS.md#4-auto-generate-symbols-without-explicit-assignment)

**Antes (v0.x):** ❌ Error si propagas sin especificar `symbol=`  
**Ahora (v1.0+):** ✅ Usa automáticamente `"_result"` como símbolo

```python
# Sin symbol=, funciona!
R = mh.quantity("V/I", "ohm")  # ← no symbol=parameter

registry = mh.register(V, I, R)
R_result = mh.propagate_quantity(R, registry)  # ✅ ¡Sin error!
```

---

## Documentos Principales

### Para LaTeX y Tablas
- **[CHANGELOG_V1_LATEXTOOLS.md](CHANGELOG_V1_LATEXTOOLS.md)** — Guía completa de cambios en latex_tools
- **[README_latex_tools.md](README_latex_tools.md)** — Sección "Experimental Data Groups" y "v1.0 Table Format Changes"

### Para Propagación de Grupos  
- **[README_uncertainties.md](README_uncertainties.md)** — Sección "Group-Aware Propagation Modes" (actualizada)
- **[README_uncertainties.md](README_uncertainties.md)** — Sección "Group Inheritance in Propagation" (nueva)

### Para Propagación en General
- **[README_uncertainties.md](README_uncertainties.md)** — Sección "Symbolic Error Propagation: `propagate_quantity()`"

---

## Matriz de Cambios

| Característica | Antes (v0.x) | Ahora (v1.0+) | Documentado en |
|---|---|---|---|
| Tablas con grupos | Manual | Automático | `README_latex_tools.md`, `CHANGELOG_V1_LATEXTOOLS.md` |
| Herencia de grupos | Solo si TODAS tienen | Si ALGUNA tiene | `README_uncertainties.md`, `CHANGELOG_V1_LATEXTOOLS.md` |
| Paréntesis en tablas | Siempre | Nunca | `CHANGELOG_V1_LATEXTOOLS.md` |
| "(1)" para dimensionless | Siempre | Nunca | `CHANGELOG_V1_LATEXTOOLS.md` |
| Símbolo requerido | Sí (error si falta) | No (default "_result") | `CHANGELOG_V1_LATEXTOOLS.md` |

---

## Quick Links

1. **Quiero tablas con grupos** → [README_latex_tools.md#automatic-group-based-tables](README_latex_tools.md#automatic-group-based-tables)

2. **Quiero heredar grupos en propagación** → [README_uncertainties.md#group-aware-propagation-modes](README_uncertainties.md#group-aware-propagation-modes)

3. **Quiero ver todos los cambios de formato** → [CHANGELOG_V1_LATEXTOOLS.md](CHANGELOG_V1_LATEXTOOLS.md)

4. **Tengo código v0.x, ¿qué cambia?** → [CHANGELOG_V1_LATEXTOOLS.md#migration-guide-from-v0x](CHANGELOG_V1_LATEXTOOLS.md#migration-guide-from-v0x)

---

## Compatibilidad

✅ **Totalmente compatible hacia atrás** con v0.x  
✅ Código existente sigue funcionando  
✅ Nuevas características son opt-in  
✅ Sin cambios breaking en la API pública

