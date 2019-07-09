#!/bin/bash
rm -rf *egg-info build dist/* && python setup.py sdist bdist_wheel
"""
https://dzone.com/articles/executable-package-pip-install
Go into your package folder and execute this command: python setup.py bdist_wheel.
This will create a  structure like this:
---- build: build package information.
---- dist: Contains your .whl file. A WHL file is a package saved in the
        Wheel format, which is the standard built-package format used for
        Python distributions. You can directly install a .whl file using
        pip install some_package.whl on your system
---- project.egg.info: An egg package contains compiled bytecode, package
        information, dependency links, and captures the info used by the
        setup.py test command when running tests.
"""
# for windows
# python setup.py bdist_wininst /or
# python setup.py bdist_wheel
# python setup.py bdist --formats=wininst