#!/bin/bash
rm -rf *egg-info build dist/* && python setup.py sdist bdist_wheel

# for windows
# python setup.py bdist_wininst /or
# python setup.py bdist --formats=wininst