#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
new_MFC setup.

"""
name = 'new_MFC'

if __name__ == '__main__':
    from setuptools import setup, find_packages


    def readme():
        with open('README.md') as f:
            return f.read()

    test_deps = ['pytest']

    # url = 'https://github.com/ashenafimenza/%s' % name

    setup(
        name=name,
        version='1.0.0',
        packages=find_packages(),
        license="European Union Public Licence 1.1 or later (EUPL 1.1+)",
        description='A lightweight microsimulation free-flow acceleration model'
                    '(MFC) that is able to capture the vehicle acceleration '
                    'dynamics accurately and consistently',
        long_description=readme(),
        project_urls={"Sources": "https://github.com/ashenafimenza/new_MFC"},
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Manufacturing',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: Implementation :: CPython',
            'Natural Language :: English',
            'Environment :: Console',
            'Operating System :: MacOS :: MacOS X',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: Unix',
        ],
        install_requires=[
            'PyYAML',
            'schedula',
            'numpy',
        ],
        tests_require=test_deps,
        package_data={
            'new_MFC': [
                'test/*.yaml',
                'db/*.csv'
            ]
        },
        include_package_data=True,
        zip_safe=True,
        options={'bdist_wheel': {'universal': True}},
        platforms=['any'],
    )
