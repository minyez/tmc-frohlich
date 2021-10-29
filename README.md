---
title: workflow script for band energy renormalization based on Frohlich model
---

## Theory

TBA

## Requirements

- Python 2.7.18 or 3.7
- NumPy (general-purpose math)
- SciPy (optimization)

## Workflow

The main script `tmc-frohlich.py` has a couple of parameters to specify.

Besides, you need the following files to run the calculation,

- `POSCAR.0`: the structure file
- `POTCAR`: the PAW potential file

### Step 0 (optional): geometry optimization

### Step 1: phonon calculation with DFPT

### Step 2: searching VBM and/or CBM

### Step 3: effective mass calculation

This step will essentially use [`emc.py`](https://github.com/afonari/emc)
written by Alexandr Fonari and Chris Sutton.
Some modifications are made in this project for the purposes:

- use as module
- remove input file dependency, use parameters of main function instead.
- Python 2/3 compatibility.
- pruned interfaces other than VASP

## TODOs

- [ ] rewrite some `emc` functions to use numpy
- [ ] optimize the execution: DFPT and VBM/CBM searching can be performed at the same time

