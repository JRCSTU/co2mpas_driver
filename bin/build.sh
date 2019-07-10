#!/bin/bash
rm -rf *egg-info build dist/* && python setup.py sdist bdist_wheel
"""
1. Ensure pip, setuptools, and wheel are up to date
    - python -m pip install --upgrade pip setuptools wheel

https://setuptools.readthedocs.io/en/latest/setuptools.html#automatic-script-creation
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
# python setup.py bdist --formats=wininst
# python setup.py bdist_wheel


#uninstall new_MFC
    -> pip uninstall package name
or f you don't know the list of all files, you can reinstall it with the --record option,
and take a look at the list this produces.

To record a list of installed files, you can use:
    -> python setup.py install --record files.txt
Once you want to uninstall you can use xargs to do the removal:
    # windows
    -> Get-Content files.txt | ForEach-Object {Remove-Item $_ -Recurse -Force}
    # linux you can use xargs to do the removal
    -> xargs rm -rf < files.txt