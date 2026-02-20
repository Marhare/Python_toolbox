import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations


def _to_symbol(var):
    """Convert var to sympy.Symbol if needed."""
    if isinstance(var, sp.Symbol):
        return var
    return sp.Symbol(str(var))


def _to_expr(expr):
    """Convert input to sympy.Expr; return None if it cannot be converted."""
    if isinstance(expr, sp.Lambda):
        return expr.expr
    if isinstance(expr, sp.Equality):
        return expr.lhs - expr.rhs
    if isinstance(expr, sp.Expr):
        return expr

    if isinstance(expr, str):
        try:
            # Parser explícito y predecible (sin multiplicación implícita)
            return parse_expr(expr, transformations=standard_transformations)
        except Exception:
            return None

    # números y otros objetos Python convertibles
    try:
        return sp.sympify(expr)
    except Exception:
        return None

def map_leaves(obj, func):
    if isinstance(obj, (list, tuple)):
        return [map_leaves(x, func) for x in obj]
    else:
        val = func(obj)
        if val is None:
            raise TypeError(f"Elemento no convertible a expresión simbólica: {obj}")
        return val
def shape_of(obj):
    """
    Return the shape of a nested list/tuple structure.
    Raise ValueError if structure is irregular.
    """
    if isinstance(obj, (list, tuple)):
        if not obj:
            return (0,)  # lista vacía
        
        subshapes = [shape_of(x) for x in obj]
        first = subshapes[0]

        # comprobar que todas las subformas coinciden
        if not all(s == first for s in subshapes):
            raise ValueError("Estructura irregular (no rectangular).")

        return (len(obj),) + first

    # hoja
    return ()

def deepness(obj):
    if isinstance(obj, (list, tuple)) and obj:
        return 1 + max(deepness(x) for x in obj)
    if isinstance(obj, (list, tuple)):
        return 1
    return 0

class Function:
    """
    Symbolic mathematical Function with lazy compilation and operator overloading.
    
    Wraps SymPy expressions and provides a callable interface with support for
    arithmetic operations. Functions are compiled lazily to numerical backends.
    
    Parameters
    ----------
    expr_str : str or sympy expression
        Mathematical expression as a string (e.g., "x**2 + 2*x + 1") or SymPy expression.
    vars : list, optional
        List of variables in order. If None, variables are sorted alphabetically.
    backend : str, default "numpy"
        Backend for lambdify compilation (e.g., "numpy", "math", "sympy").
    
    Examples
    --------
    >>> f = Function("x**2 + 1")
    >>> f(2)
    5
    >>> g = f + Function("x")
    >>> g(2)
    7
    """
    
    def __init__(self, expr, *args,vars=None, params= None, backend="numpy", symbol=None, indices=None):
        # 1) parse
        
        if isinstance(expr, (list, tuple)):
            expr = map_leaves(expr, _to_expr)
            shape = shape_of(expr)
            if len(shape) <= 2:
                expr = sp.Matrix(expr)
            else:
                expr = sp.Array(expr)


            
        if not isinstance(expr, (sp.Expr, sp.MatrixBase, sp.Array)):
            expr = _to_expr(expr)  # usa tu base
        if expr is None:
            raise TypeError("Function needs symbolic expression or str")
        self.expr = expr
        if isinstance(expr, sp.MatrixBase):
            self.shape = expr.shape
        elif isinstance(expr, sp.Array):
            self.shape = expr.shape
        else:
            self.shape = ()
        free = sorted(list(self.expr.free_symbols), key=lambda s:s.name)
        # 2) variables
        if vars is None:
            if len(free) <= 1:
                self.vars = free
            else:
                raise ValueError(
                    "Expresion with >1 free symbols: specify vars=[...]."
                    f"Free symbols: {[s.name for s in free]}"
                )
        else:
            self.vars = [_to_symbol(v) for v in vars]
        
        #Parameters
        if params is None:
            self.params = [s for s in free if s not in self.vars]
        else:
            self.params = [_to_symbol(p) for p in params]

        # 3) Compilation
        self.backend = backend
        self._call = None
        self.symbol= symbol
        self.indices=indices

    # --- Representation and conversion ---
    def _sympy_(self):
        """Return the underlying SymPy expression."""
        return self.expr
    
    
    def _latex_indexed(self):
        rank = len(self.shape)

        # letras estándar para índices
        default_indices = ["i","j","k","l","m","n"]

        idx = default_indices[:rank]

        if self.indices:
            lower = []
            upper = []
            for i,t in enumerate(self.indices):
                if t == "cov":
                    lower.append(idx[i])
                elif t == "contra":
                    upper.append(idx[i])
            sub = "".join(lower)
            sup = "".join(upper)
        else:
            sub = "".join(idx)
            sup = ""

        if sup and sub:
            return f"{self.symbol}^{{{sup}}}_{{{sub}}}"
        elif sup:
            return f"{self.symbol}^{{{sup}}}"
        elif sub:
            return f"{self.symbol}_{{{sub}}}"
        else:
            return self.symbol

    
    
    
    def latex(self, mode="auto"):
        """
        mode:
            - "auto": si hay symbol -> indexado, si no -> expr
            - "expr": siempre expr explícita
            - "indexed": siempre indexado (requiere symbol)
        """

        if mode == "expr":
            return sp.latex(self.expr)

        if mode == "indexed":
            if self.symbol is None:
                raise ValueError("Indexed mode requires a symbol.")
            return self._latex_indexed()

        if mode == "auto":
            if self.symbol is None:
                return sp.latex(self.expr)
            return self._latex_indexed()

        raise ValueError("mode must be 'auto', 'expr', or 'indexed'")


    # --- Evaluation ---
    def compile(self):
        """Compile the expression to a callable Function using the specified backend."""
        if self._call is None:
            self._call = sp.lambdify(self.vars+self.params, self.expr, self.backend)
        return self._call

    def __call__(self, *args, **kwargs):
        """
        Evaluate the Function at given values.
        
        Parameters
        ----------
        *args : float or array-like
            Positional arguments in the order of variables.
        **kwargs : dict
            Named arguments mapping variable names or symbols to values.
            
        Returns
        -------
        float or array
            Evaluated result.
        """
         # Composición funcional
        if len(args) == 1 and isinstance(args[0], Function):
            g = args[0]

            if len(self.vars) != 1:
                raise ValueError("Composition doesn't support for unidimension Functions")

            # Sustituimos variable de f por la expresión de g
            new_expr = self.expr.subs(self.vars[0], g.expr)

            # Variables nuevas = unión ordenada
            new_vars = sorted(list(new_expr.free_symbols), key=lambda s: s.name)

            return Function(new_expr, vars=new_vars, backend=self.backend)
            
        
        
        f = self.compile()

        if kwargs:
            ordered = []
            for s in (self.vars+self.params):
                key = str(s)
                if key in kwargs:
                    ordered.append(kwargs[key])
                elif s in kwargs:
                    ordered.append(kwargs[s])
                else:
                    raise ValueError(f"Missing value for {s}")
            return f(*ordered)

        return f(*args)

    # --- Operation helpers ---
    def _wrap(self, new_expr):
        """Wrap a new expression into a Function with updated variables and backend."""
        new_vars= sorted(list(new_expr.free_symbols), key = lambda s: s.name)
        return Function(new_expr, vars=new_vars, backend=self.backend)

    def _as_expr(self, other):
        """Convert other to SymPy expression if needed."""
        return other.expr if isinstance(other, Function) else sp.sympify(other)

    def _binop(self, other, op):
        """Apply a binary operation and wrap the result."""
        return self._wrap(op(self.expr, self._as_expr(other)))

    def _rbinop(self, other, op):
        """Apply a reverse binary operation and wrap the result."""
        return self._wrap(op(self._as_expr(other), self.expr))

    # --- Operators ---
    # Suma
    def __add__(self, other):
        return self._binop(other, lambda a,b: a+b)

    def __radd__(self, other):
        return self._rbinop(other, lambda a,b: a+b)

    # Resta
    def __sub__(self, other):
        return self._binop(other, lambda a,b: a-b)

    def __rsub__(self, other):
        return self._rbinop(other, lambda a,b: a-b)

    # Multiplicación
    def __mul__(self, other):
        return self._binop(other, lambda a,b: a*b)

    def __rmul__(self, other):
        return self._rbinop(other, lambda a,b: a*b)

    # División
    def __truediv__(self, other):
        return self._binop(other, lambda a,b: a/b)

    def __rtruediv__(self, other):
        return self._rbinop(other, lambda a,b: a/b)

    # Potencia
    def __pow__(self, other):
        return self._binop(other, lambda a,b: a**b)

    def __rpow__(self, other):
        return self._rbinop(other, lambda a,b: a**b)

    # Negativo
    def __neg__(self):
        return self._wrap(-self.expr)
    

#Special Operators
def dt(f, var):
    if not isinstance(f, Function):
        raise TypeError("total espera un objeto Function.")

    v = sp.Symbol(var) if isinstance(var, str) else var

    # SymPy aplica automáticamente regla de la cadena
    expr = sp.diff(f.expr, v)

    return Function(expr, *f.vars,
                    params=f.params,
                    backend=f.backend)


def dp(f, var):
    if not isinstance(f, Function):
        raise TypeError("partial espera un objeto Function.")

    v = sp.Symbol(var) if isinstance(var, str) else var
    expr = sp.diff(f.expr, v)

    return Function(expr, *f.vars,
                    params=f.params,
                    backend=f.backend)

def I(f, var=None, interval=None):
    """
    Integration operator.
    
    Computes the indefinite or definite integral of a Function.
    
    Parameters
    ----------
    f : Function
        The Function to integrate.
    var : str or sympy Symbol, optional
        Variable of integration. If None, uses the first variable.
    interval : tuple of (a, b), optional
        If provided, computes the definite integral from a to b.
        If None, computes the indefinite integral.
        
    Returns
    -------
    Function or float/array
        For indefinite integral: a new Function object.
        For definite integral: the numerical result.
        
    Examples
    --------
    >>> f = Function("x")
    >>> F = I(f, "x")  # indefinite
    >>> F.expr  # x**2/2
    
    >>> result = I(f, "x", interval=(0, 1))  # definite
    >>> result
    0.5
    """
    if not isinstance(f, Function):
        raise TypeError("I espera un objeto Function.")

    # Determinar variable
    if var is None:
        if not f.vars:
            raise ValueError("La función no tiene variables.")
        v = f.vars[0]
    else:
        v = sp.Symbol(var) if isinstance(var, str) else var

    # Integral indefinida
    if interval is None:
        new_expr = sp.integrate(f.expr, v)
        return Function(new_expr, backend=f.backend)

    # Integral definida
    if not (isinstance(interval, tuple) and len(interval) == 2):
        raise ValueError("interval debe ser una tupla (a,b)")

    a, b = interval
    result = sp.integrate(f.expr, (v, a, b))

    return result

def jacobian(f, vars=None):
    if not isinstance(f, Function):
        raise TypeError("It is needed a Function object to compute the Jacobian")
    if not f.vars:
        raise TypeError("There are no variables to derive respect")
    #If scalar -> (1,n)
    if f.shape == ():
        row = [sp.diff(f.expr, v) for v in f.vars]
        J = sp.Matrix([row])
        return Function(J,*f.vars,params=f.params, backend=f.backend)
    
    #If column vector (m,1)
    if isinstance(f.expr,sp.MatrixBase):
        J = f.expr.jacobian(f.vars)
        return Function(J, *f.vars, params=f.params, backend=f.backend)


# Alias for backward compatibility with documentation
D = dt



