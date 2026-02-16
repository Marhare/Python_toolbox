import sympy as sp


def _to_symbol(var):
    """Convert var to sympy.Symbol if needed."""
    if isinstance(var, sp.Symbol):
        return var
    return sp.Symbol(str(var))


def _to_expr(expr):
    """Convert expr to sympy.Expr when possible; None if callable."""
    if isinstance(expr, sp.Lambda):
        return expr.expr
    if isinstance(expr, sp.Expr):
        return expr
    if isinstance(expr, sp.Equality):
        return expr.lhs - expr.rhs
    if isinstance(expr, str):
        try:
            return sp.sympify(expr)
        except Exception:
            return None
    return None

class Function:
    """
    Symbolic mathematical function with lazy compilation and operator overloading.
    
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
    
    def __init__(self, expr_str: str, *, vars=None, backend="numpy"):
        # 1) parse
        expr = _to_expr(expr_str)  # usa tu base
        if expr is None:
            raise TypeError("Function espera un str o una expresión simbólica.")
        self.expr = expr

        # 2) variables
        if vars is None:
            self.vars = sorted(list(self.expr.free_symbols), key=lambda s: s.name)
        else:
            self.vars = [_to_symbol(v) for v in vars]

        # 3) compilación perezosa
        self.backend = backend
        self._call = None

    # --- Representation and conversion ---
    def _sympy_(self):
        """Return the underlying SymPy expression."""
        return self.expr

    def latex(self):
        """Return LaTeX representation of the function."""
        return sp.latex(self.expr)

    # --- Evaluation ---
    def compile(self):
        """Compile the expression to a callable function using the specified backend."""
        if self._call is None:
            self._call = sp.lambdify(self.vars, self.expr, self.backend)
        return self._call

    def __call__(self, *args, **kwargs):
        """
        Evaluate the function at given values.
        
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
                raise ValueError("Composición solo soportada para funciones univariantes.")

            # Sustituimos variable de f por la expresión de g
            new_expr = self.expr.subs(self.vars[0], g.expr)

            # Variables nuevas = unión ordenada
            new_vars = sorted(list(new_expr.free_symbols), key=lambda s: s.name)

            return Function(new_expr, vars=new_vars, backend=self.backend)
            
        
        
        f = self.compile()

        if kwargs:
            values = []
            for s in self.vars:
                key = str(s)
                if key in kwargs:
                    values.append(kwargs[key])
                elif s in kwargs:
                    values.append(kwargs[s])
                else:
                    raise ValueError(f"Falta valor para variable {s}")
            return f(*values)

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
def D(f, *vars):
    """
    Derivative operator.
    
    Computes the derivative of a Function with respect to one or more variables.
    
    Parameters
    ----------
    f : Function
        The function to differentiate.
    *vars : str or sympy Symbol
        Variables to differentiate with respect to, in order.
        
    Returns
    -------
    Function
        The derivative as a new Function object.
        
    Examples
    --------
    >>> f = Function("x**2")
    >>> df = D(f, "x")
    >>> df(3)
    6
    """
    if not isinstance(f, Function):
        raise TypeError("D espera Function.")

    expr = f.expr

    for v in vars:
        v = sp.Symbol(v) if isinstance(v, str) else v
        expr = sp.diff(expr, v)

    return Function(expr)

def I(f, var=None, interval=None):
    """
    Integration operator.
    
    Computes the indefinite or definite integral of a Function.
    
    Parameters
    ----------
    f : Function
        The function to integrate.
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



