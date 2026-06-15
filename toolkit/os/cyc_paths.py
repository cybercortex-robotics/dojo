"""
Copyright (c) RovisLab
RovisDojo: RovisLab neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing,
please contact Prof. Sorin Grigorescu (s.grigorescu@unitbv.ro)
"""

import os
import sys


def find_dojo_parent(start_dir):
    """
    Walk up from start_dir to find the ancestor that actually contains a
    usable 'dojo' package. This is the repo itself when run as its own repo
    (where 'dojo' is its own submodule), or a parent project's root when this
    repo is used as a submodule (where 'dojo' lives at the parent root instead).
    """
    current = os.path.abspath(start_dir)
    while True:
        if os.path.isdir(os.path.join(current, "dojo", "data")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def add_dojo_parent_to_path(start_dir):
    """Find the ancestor with a usable 'dojo' package and append it to sys.path."""
    dojo_parent = find_dojo_parent(start_dir)
    if dojo_parent is not None:
        sys.path.append(dojo_parent)
    return dojo_parent
