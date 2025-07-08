"""
The module describes calculators - objects that 
receive :class:`~wannierberri.data_K._Data_K` objects and yield
:class:`~wannierberri.result.Result`
"""

from .calculator import Calculator
from . import static, dynamic, tabulate, sdct, q2
from .tabulate import TabulatorAll
