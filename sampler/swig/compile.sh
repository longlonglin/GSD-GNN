#!/bin/bash

swig -c++ -python -py3 selector.i

python3 setup.py build_ext --inplace