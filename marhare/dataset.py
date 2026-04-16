"""Tabular container for aligned experimental data.

``Dataset`` stores columns as either:

- ``Quantity`` objects for measured variables with uncertainty,
- plain ``numpy`` arrays for categorical/metadata columns.

It provides shape alignment checks, row filtering, grouping, and conversion to/from
``pandas.DataFrame``.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union

# Import Quantity from computation layer
from marhare.quantities import Quantity

class Dataset:
    """A labeled collection of row-aligned experimental variables."""
    __slots__ = ("name", "metadata", "_data", "_nrows")

    def __init__(self, data: Optional[Dict[str, Any]] = None, name: str = "Experiment", metadata: Optional[dict] = None):
        self.name = name
        self.metadata = metadata or {}
        self._data: Dict[str, Union[Quantity, np.ndarray]] = {}
        self._nrows: int = 0
        if data:
            for key, val in data.items():
                self[key] = val

    @property
    def nrows(self) -> int: return self._nrows
    
    @property
    def columns(self) -> List[str]: return list(self._data.keys())

    def __getitem__(self, key: str) -> Union[Quantity, np.ndarray]:
        if key not in self._data: raise KeyError(f"Column '{key}' not found.")
        return self._data[key]

    def __setitem__(self, key: str, value: Any):
        """
        Smart insertion: 
        - Quantities pass through.
        - Numeric arrays become unitless Quantities.
        - Strings/Booleans become standard numpy arrays.
        - Scalars are broadcasted to match dataset rows.
        """
        if isinstance(value, Quantity):
            processed_val = value
            n_val = len(processed_val.value)
        else:
            arr = np.asanyarray(value)
            
            # Broadcast scalar to match existing rows
            if arr.ndim == 0 and self._nrows > 0:
                arr = np.repeat(arr, self._nrows)
                
            n_val = len(arr) if arr.ndim > 0 else 1
            
            # Check if data is numeric
            if np.issubdtype(arr.dtype, np.number):
                processed_val = Quantity(arr, np.zeros_like(arr), unit="1", symbol=key, traceable=False)
            else:
                # Categorical data (strings, bools, objects) stays as raw array
                processed_val = arr

        # Alignment Check
        if not self._data:
            self._nrows = n_val
        elif n_val != self._nrows:
            raise ValueError(f"Alignment error: '{key}' has {n_val} rows, but Dataset expects {self._nrows}.")
        
        self._data[key] = processed_val

    def select(self, mask: Union[np.ndarray, list]) -> 'Dataset':
        """Filters both Quantities and categorical arrays."""
        mask = np.asarray(mask, dtype=bool)
        if len(mask) != self.nrows:
            raise ValueError("Mask length must match Dataset rows.")

        # Preserve original column types (Quantity vs raw ndarray) instead of
        # round-tripping through __setitem__, which may coerce numeric arrays.
        out = Dataset(name=f"{self.name}_filtered", metadata=self.metadata.copy())
        out._nrows = int(mask.sum())

        for key, val in self._data.items():
            if isinstance(val, Quantity):
                out._data[key] = val[mask]
            else:
                out._data[key] = np.asarray(val)[mask].copy()

        return out

    def where(self, **conds) -> 'Dataset':
        """Filter rows by exact-match conditions on one or more columns.

        Examples
        --------
        >>> ds.where(Voltage=3100, Tag1="anillo_grande")
        """
        mask = np.ones(self.nrows, dtype=bool)

        def _normalize_condition(
            condition_value: Any,
            column_name: str,
            expected_shape: tuple,
        ):
            """Normalize user conditions for exact/row-aligned filtering.

            Accepted forms:
            - scalar,
            - singleton array (size 1),
            - row-aligned array with the same shape as the target column.
            """
            arr = np.asarray(condition_value)
            if arr.shape == ():
                return arr.item()
            if arr.size == 1:
                return np.asarray(arr.reshape(-1)[0]).item()
            if arr.shape == expected_shape:
                return arr
            raise TypeError(
                f"Condition for column '{column_name}' must be scalar or "
                f"row-aligned with shape {expected_shape}, got shape {arr.shape}."
            )

        for key, val in conds.items():
            if key not in self._data:
                raise KeyError(f"Column '{key}' not found.")

            col = self[key]
            if isinstance(col, Quantity):
                raw = np.asarray(col.value)
                if isinstance(val, Quantity):
                    val = val.to(col.unit, normalize=False).value
                val = _normalize_condition(val, key, raw.shape)
            else:
                raw = np.asarray(col)
                if isinstance(val, Quantity):
                    raise TypeError(
                        f"Condition for non-quantity column '{key}' cannot be a Quantity."
                    )
                val = _normalize_condition(val, key, raw.shape)

            mask &= (raw == val)

        return self.select(mask)

    def filter_rows(self, **conds) -> 'Dataset':
        """Alias for :meth:`where` kept for readability in notebooks."""
        return self.where(**conds)

    def get_quantity(self, column: str, **conds) -> Quantity:
        """Return a single-row Quantity selected by conditions.

        Raises
        ------
        KeyError
            If no rows match conditions.
        ValueError
            If multiple rows match conditions.
        TypeError
            If the requested column is not a Quantity column.
        """
        sub = self.where(**conds)

        if sub.nrows == 0:
            raise KeyError(conds)
        if sub.nrows > 1:
            raise ValueError(f"Multiple rows match {conds}")

        out = sub[column]
        if not isinstance(out, Quantity):
            raise TypeError(f"Column '{column}' is not a Quantity column.")

        out_value = np.asarray(out.value)
        if out_value.shape == ():
            return out

        # One-row selection: return scalar quantity for ergonomic downstream math.
        return out[0]

    def group_by(self, column: str) -> Dict[Any, 'Dataset']:
        """Groups by unique values in either a Quantity or a categorical column."""
        col_data = self[column]
        raw_vals = col_data.value if isinstance(col_data, Quantity) else col_data
        unique_vals = np.unique(raw_vals)
        
        groups = {}
        for val in unique_vals:
            mask = (raw_vals == val)
            groups[val] = self.select(mask)
        return groups

    def to_pandas(self) -> pd.DataFrame:
        df_dict = {}
        for col, v in self._data.items():
            if isinstance(v, Quantity):
                df_dict[col] = v.value
                if np.any(v.sigma > 0): df_dict[f"{col}_sigma"] = v.sigma
            else:
                df_dict[col] = v
        return pd.DataFrame(df_dict)

    @classmethod
    def from_pandas(cls, df: pd.DataFrame, name: str = "Imported") -> 'Dataset':
        ds = cls(name=name)
        processed_cols = set()
        
        for col in df.columns:
            if col in processed_cols or col.endswith("_sigma"): 
                continue
            
            # 1. Datos categóricos / texto
            if not np.issubdtype(df[col].dtype, np.number):
                ds[col] = df[col].values
                processed_cols.add(col)
                continue
                
            # 2. Extracción de Unidad con Expresión Regular: "voltaje (V)" -> symbol="voltaje", unit="V"
            match = re.match(r"^(.*?)\s*[\(\[](.*?)[\)\]]$", str(col))
            if match:
                symbol = match.group(1).strip()
                unit = match.group(2).strip()
            else:
                symbol = str(col).strip()
                unit = "1"
                
            # 3. Buscar la columna de error (_sigma)
            possible_sigma_names = [
                f"{col}_sigma",               
                f"{symbol}_sigma",            
                f"{symbol}_sigma ({unit})",   
                f"{symbol}_sigma [{unit}]"    
            ]
            
            sigma_col = None
            for p_name in possible_sigma_names:
                if p_name in df.columns:
                    sigma_col = p_name
                    break
            
            # 4. Extraer arrays y crear la Quantity
            val_array = df[col].values
            sig_array = df[sigma_col].values if sigma_col else np.zeros_like(val_array)
            
            ds[symbol] = Quantity(val_array, sig_array, unit=unit, symbol=symbol)
            
            processed_cols.add(col)
            if sigma_col: 
                processed_cols.add(sigma_col)
            
        return ds

    @classmethod
    def from_measurement_table(
        cls,
        df: pd.DataFrame,
        *,
        value_col: str = "Value",
        sigma_col: str = "Uncertainty",
        unit_col: str = "Unit",
        quantity_name: str = "q",
        name: str = "MeasurementTable",
    ) -> 'Dataset':
        """Build Dataset from long-format measurement tables.

        Expected input shape:
            metadata columns + value/sigma/unit columns.

        Output:
            - metadata columns kept as plain numpy arrays,
            - one quantity column named ``quantity_name``.
        """
        required = [value_col, sigma_col, unit_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

        ds = cls(name=name)
        meta_cols = [c for c in df.columns if c not in (value_col, sigma_col, unit_col)]

        ds._nrows = len(df)
        for col in meta_cols:
            ds._data[col] = np.asarray(df[col])

        values = np.asarray(df[value_col])
        sigmas = np.asarray(df[sigma_col])
        units_arr = np.asarray(df[unit_col])

        if len(values) == 0:
            ds._data[quantity_name] = Quantity(
                np.array([], dtype=float),
                np.array([], dtype=float),
                unit="1",
                symbol=quantity_name,
                traceable=False,
            )
            return ds

        q_rows = [
            Quantity(v, s, str(u), symbol=quantity_name)
            for v, s, u in zip(values, sigmas, units_arr)
        ]

        base_unit = q_rows[0].unit
        vals = []
        sigs = []
        for i, q in enumerate(q_rows):
            q_conv = q
            if q.unit != base_unit:
                try:
                    q_conv = q.to(base_unit, normalize=False)
                except Exception as exc:
                    raise ValueError(
                        f"Row {i}: unit '{q.unit}' is not compatible with '{base_unit}'"
                    ) from exc

            vals.append(float(np.asarray(q_conv.value, dtype=float)))
            sigs.append(float(np.asarray(q_conv.sigma, dtype=float)))

        ds._data[quantity_name] = Quantity(
            np.asarray(vals, dtype=float),
            np.asarray(sigs, dtype=float),
            unit=base_unit,
            symbol=quantity_name,
            traceable=False,
        )

        return ds

    @classmethod
    def from_long_table(
        cls,
        df: pd.DataFrame,
        *,
        value_col: str = "Value",
        sigma_col: str = "Uncertainty",
        unit_col: str = "Unit",
        quantity_col: str = "q",
        name: str = "MeasurementTable",
    ) -> 'Dataset':
        """Alias for :meth:`from_measurement_table` for explicit naming."""
        return cls.from_measurement_table(
            df,
            value_col=value_col,
            sigma_col=sigma_col,
            unit_col=unit_col,
            quantity_name=quantity_col,
            name=name,
        )

    @classmethod
    def read_csv(cls, filepath: str, name: str = "Imported_CSV", **kwargs) -> 'Dataset':
        """Shortcut to read a CSV directly into a Dataset."""
        df = pd.read_csv(filepath, **kwargs)
        return cls.from_pandas(df, name=name)

    @classmethod
    def read_excel(cls, filepath: str, name: str = "Imported_Excel", **kwargs) -> 'Dataset':
        """Shortcut to read an Excel file directly into a Dataset."""
        df = pd.read_excel(filepath, **kwargs)
        return cls.from_pandas(df, name=name)

    def __repr__(self):
        return f"<Dataset '{self.name}': {self.nrows} rows, columns={self.columns}>"
