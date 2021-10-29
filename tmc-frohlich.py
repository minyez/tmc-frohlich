#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""a script to compute band energy renormalization due to electron-phonon coupling
based on Frohlich model, interfaced with VASP.

See README.md for illustration of the workflow.
"""
from __future__ import print_function
from argparse import ArgumentParser, RawDescriptionHelpFormatter

__version__ = '0.0.1'

def _parser():
    p = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    p.add_argument('--stc', dest="stencil", default=3, choices=[3, 5], type=int,
                   help="<5> stencil")
    p.add_argument('-h', dest="stepsize", default=0.1, type=float,
                   help="<0.1> stepsize of finite difference")
    return p

if __name__ == "__main__":
    args = _parser().parse_args()
