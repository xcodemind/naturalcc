#!/usr/bin/python  
#python version: 2.7.3  
#Filename: SetupTestOMP.py  
   #set(PYTHON_INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS} /usr/local/lib/python3.5/dist-packages/numpy/core/include)

# Run as:    
#    python setup.py build_ext --inplace    
     
import sys    
sys.path.insert(0, "..")    
     
from distutils.core import setup    
from distutils.extension import Extension    
from Cython.Build import cythonize
import numpy
from Cython.Distutils import build_ext  
     
# ext_module = cythonize("TestOMP.pyx")    
# ext_module = Extension(  
#                         "data_utils_fast",  
#             ["data_utils_fast.pyx"],  
#             extra_compile_args=["/openmp"],  
#             extra_link_args=["/openmp"],  
#             )  
     
# setup(  
#     cmdclass = {'build_ext': build_ext},  
#         ext_modules = [ext_module],   
# )
# setup(
#     ext_modules=[
#         Extension("data_utils_fast", ["my_module.c"],
#                   include_dirs=[numpy.get_include()]),
#     ],
# )

setup(
    ext_modules = cythonize("data_utils_fast.pyx"),
    include_dirs=[numpy.get_include()]
)