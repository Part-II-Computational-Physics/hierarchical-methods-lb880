"""
Barnes-Hut (BH)
===============
Code for the Barnes-Hut algorithm for N-Body approximation.

Algorithm as described in 'A hierarchical O(N log N) force-calculation
algorithm' by Barnes and Hut.
Made in reference to https://github.com/barbagroup/FMM_tutorial. 
"""

from . import cells
from .cells import Cell, RootCell

from . import bh
from .bh import *