#!/usr/bin/env python3
# Copyright (c) HUST, UTS and ZJU.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from setuptools import setup, find_packages, Extension
import sys

if sys.version_info < (3, 6):
    sys.exit('Sorry, Python >= 3.6 is required for natural code.')

with open('README.md', encoding='UTF-8') as f:
    readme = f.read()

if sys.platform == 'darwin':
    extra_compile_args = ['-stdlib=libc++', '-O3']
else:
    extra_compile_args = ['-std=c++11', '-O3']


class NumpyExtension(Extension):
    """Source: https://stackoverflow.com/a/54128391"""

    def __init__(self, *args, **kwargs):
        self.__include_dirs = []
        super().__init__(*args, **kwargs)

    @property
    def include_dirs(self):
        import numpy
        return self.__include_dirs + [numpy.get_include()]

    @include_dirs.setter
    def include_dirs(self, dirs):
        self.__include_dirs = dirs


extensions = [
    Extension(
        'ncc.libbleu',
        sources=[
            'ncc/clib/libbleu/libbleu.cpp',
            'ncc/clib/libbleu/module.cpp',
        ],
        extra_compile_args=extra_compile_args,
    ),
    NumpyExtension(
        'ncc.data.tools.data_utils_fast',
        sources=['ncc/data/tools/data_utils_fast.pyx'],
        language='c++',
        extra_compile_args=extra_compile_args,
    ),
    NumpyExtension(
        'ncc.data.tools.token_block_utils_fast',
        sources=['ncc/data/tools/token_block_utils_fast.pyx'],
        language='c++',
        extra_compile_args=extra_compile_args,
    ),
]

cmdclass = {}

try:
    # torch is not available when generating docs
    from torch.utils import cpp_extension

    extensions.extend([
        cpp_extension.CppExtension(
            'ncc.libnat',
            sources=[
                'ncc/clib/libnat/edit_dist.cpp',
            ],
        )
    ])

    if 'CUDA_HOME' in os.environ:
        extensions.extend([
            cpp_extension.CppExtension(
                'ncc.libnat_cuda',
                sources=[
                    'ncc/clib/libnat_cuda/edit_dist.cu',
                    'ncc/clib/libnat_cuda/binding.cpp'
                ],
            )])
    cmdclass['build_ext'] = cpp_extension.BuildExtension

except ImportError:
    pass

if 'clean' in sys.argv[1:]:
    # Source: https://bit.ly/2NLVsgE
    print("deleting Cython files...")
    import subprocess

    subprocess.run(['rm -f ncc/*.so ncc/**/*.so ncc/*.pyd ncc/**/*.pyd'], shell=True)

setup(
    name='ncc',
    version='0.4.0',
    description='NaturalCode: A Benchmark towards Understanding theNaturalness of Source Code and Comment',
    url='https://github.com/whatsmyname/naturalcodev3',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence:: Software Engineering',
    ],
    long_description=readme,
    long_description_content_type='text/markdown',
    setup_requires=[
        'cython',
        'numpy',
        'setuptools>=18.0',
    ],
    install_requires=[
        'cffi',
        'cython',
        'numpy',
        'regex',
        'sacrebleu',
        'torch',
        'tqdm',
    ],
    # dependency_links=dependency_links,
    packages=find_packages(exclude=['scripts', 'vis', 'web', 'run', 'exp', 'exp', 'doc']),
    ext_modules=extensions,
    # test_suite='tests',
    entry_points={
        'console_scripts': [
            'ncc-run = run.main:main',
        ],
    },
    cmdclass=cmdclass,
    zip_safe=False,
)
