#!/bin/bash

rm -r build

rm _walker.cpython-38-x86_64-linux-gnu.so walker_wrap.cxx walker.py 

swig -c++ -python -py3 walker.i

python3 setup.py build_ext --inplace