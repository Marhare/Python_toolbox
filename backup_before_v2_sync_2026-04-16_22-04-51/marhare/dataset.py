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
        
        new_data = {k: v[mask] for k, v in self._data.items()}
        return Dataset(new_data, name=f"{self.name}_filtered", metadata=self.metadata.copy())

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
