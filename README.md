# Workflow script for band energy renormalization based on Frohlich model

## Theory

TBA

## Requirements

- Python 2.7.18 or 3.7
- NumPy (general-purpose math)
- SciPy (optimization routines for band edge search)

## Usage

Put the main script `tmc-frohlich.py` and `emc.py` module to your working directory,
e.g. `/path/to/workdir`.

```shell
cp tmc-frohlich.py emc.py /path/to/workdir
cd /path/to/workdir
```

Then add the following files required by VASP to run the calculations

- `POSCAR`: the structure file
- `POTCAR`: the PAW potential file
- `KPOINTS`: the kmesh that will be used for SCF/DFPT

That's all. If you already have `mpirun` and `vasp_std` in `PATH`,
you can start the calculation by issueing

```shell
python tmc-frohlich.py -n 4
```

where `-n` asks the computer to run 4 parallel tasks. This will run the full [workflow](#Workflow).

Note that the main script `tmc-frohlich.py` has a couple of parameters to specify.
In particular, you can create an equally-spaced Gamma-centered kmesh by
specifying the `-k` option, such that you don't have to write `KPOINTS` beforehand.

For more details about the options, try

```shell
python tmc-frohlich.py -h
```

## Workflow

### Step 1: SCF with optional geometry optimization

This step is mainly to provide a converged CHGCAR for the later effective mass calculation.

This step also allows a geometry optimization which can be switched on by `--opt`.

### Step 2: calculate dielectric tensor by DFPT with phonon contribution included

### Step 3: compute effective mass

This step will try to compute the electron effective mass at the band edge.
The location of the band edges, i.e. valence band maximum (VBM) and conduction band minimum (CBM),
are specified by `--kv` and `--kv` argument, respectively.
If not, the script will search the whole BZ by optimization function implemented in SciPy.

For effective mass calculation, it will essentially use [`emc.py`](https://github.com/afonari/emc)
written by Alexandr Fonari and Chris Sutton.
Some modifications are made in this project for the purposes:

- use as module
- remove input file dependency and instead use parameters of the main function
- Python 2/3 compatibility
- pruned interfaces other than VASP

## Examples

NaCl (both VBM and CBM at Gamma point) with PBEsol functional and geometry optimization.
Five-point stencil and step size of 0.05 inverse Bohr is used.

```shell
python tmc-frohlich.py -n 4 -k 8 8 8 --opt --xc pbesol --kv 0 0 0 --kc 0 0 0 --stc 5 -s 0.05
```

## TODOs

- [ ] rewrite some `emc` functions to use numpy (WIP)
- [ ] optimize the execution: DFPT and VBM/CBM searching can be performed at the same time
- [ ] support for spin-polarization, i.e. ISPIN=2 (?)

## Known issues

- [x] Frohlich EPC renormalization applies only to polar materials. For elementary substances, the ionic contribution to dielectric tensor vanishes.

