"""
Fast Multipole Method
=====================
Code for the Fast Multipole Method for N-Body approximation.

Algorithm as described in 'A Fast Algorithm for Particle Simulations'
by Greengard and Rokhlin.
"""

from .fmm import *
from .level import *
from .cell import *

from . import cell
from . import level
from . import fmm

from .. import fmm