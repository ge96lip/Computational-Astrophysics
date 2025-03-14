from distutils.core import setup
from Cython.Build import cythonize
import numpy


setup(ext_modules=cythonize("cythonfn_pl.pyx", 
                            compiler_directives={"language_level":"3", 'boundscheck':'False'}),include_dirs=[numpy.get_include()])