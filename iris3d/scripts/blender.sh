#!/usr/bin/env bash
# Run Blender with numpy available to its embedded Python.
# Set BLENDER_PYLIB to a folder holding numpy built for Blender's Python
# (default: /opt/blender-pylib). Official blender.org builds already ship numpy.
export PYTHONPATH="${BLENDER_PYLIB:-/opt/blender-pylib}${PYTHONPATH:+:$PYTHONPATH}"
exec "${BLENDER:-blender}" "$@"
