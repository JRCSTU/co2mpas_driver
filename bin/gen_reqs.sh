#!/bin/bash
pip freeze |sed '/^-e / s/^/#/'
