"""
Packages
========
Collection of packages for the algorithms developed within this project. 

Pairwise, Barnes-Hut, and Fast Multipole Method are described by various
classes which, when initialised, will have a `.do_method()` function to run the
algorithm with the initialised particles and parameters. 
"""

from . import general
from . import bh
from . import fmm