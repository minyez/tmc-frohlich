#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=W0707
"""a script to compute band energy renormalization due to electron-phonon coupling
based on Frohlich model, interfaced with VASP.

See README.md for illustration of the workflow.
"""
from __future__ import print_function
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import os
from shutil import copy2
from re import match
import numpy as np

from emc import emc_setup, emc_analyze, EV2H

__version__ = "0.1.0"
__author__ = "Min-Ye Zhang, Zhi-Hao Cui"

# conversion coefficient of THz to Hartree
THZ2HA = 1.519829846e-4
HA2EV = 1.0 / EV2H

def _which(exe):
    """check if executable exists"""
    try:
        from shutil import which
        path = which(exe)
    except ImportError:
        path = os.popen("which " + exe).read().strip()
        if not path:
            path = None
    return path

def get_bande_from_eigenval(ib, ispin, eigenfile='EIGENVAL'):
    """obtain the band energies of band index ib from VASP EIGENVAL file

    Args:
        ib (int): band index, starting from 1
        ispin (int)
        eigenfil (str): the path to the EIGENVAL file

    Caveat:
        Spin-polarization is not supported.
        When ispin=2, only the first spin channel is returned.
    """
    with open(eigenfile, 'r') as h:
        lines = h.readlines()
    nelect, nkpts, nbands = map(int, lines[5].split())
    if ib > nbands:
        raise ValueError("too large band index: %d > NBANDS %d" % (ib, nbands))
    lines = lines[7:]
    # TODO handle spin-polarization
    if ispin == 2:
        raise NotImplementedError
    return [float(x.split()[1]) for x in lines[ib::nbands+2]]

def check_prereq(mpi, vasp, kpts):
    """check prerequisite files"""
    for f in ["POSCAR", "POTCAR"]:
        if os.path.exists(f):
            print("%s found" % f)
        else:
            raise IOError("require %s" % f)
    if _which(mpi) is None:
        raise ValueError("mpi program is not found: %s" % mpi)
    if _which(vasp) is None:
        raise ValueError("vasp program is not found: %s" % vasp)
    if os.path.exists('KPOINTS'):
        print("KPOINTS found")
    else:
        print("KPOINTS not found, will read from arguments")
        if kpts is None:
            raise ValueError("kpts not specified")
        try:
            with open("KPOINTS", 'w') as h:
                h.write("KPOINTS by tmc-frohlich\n")
                h.write("0\nG\n")
                h.write("%d %d %d\n" % tuple(kpts))
                h.write("0 0 0\n")
            print("KPOINTS written")
        except TypeError as err:
            raise TypeError("Invalid kpoint: %r" % kpts)

def run_vasp(mpi, nproc, vasp, out='vasp.out', mode='w'):
    """run the vasp workhorse"""
    import subprocess as sp
    cmds = [mpi, '-n', str(nproc), vasp]
    with open(out, mode) as h:
        p = sp.Popen(cmds, stdout=h)
        p.wait()

def _format_common_incar_lines(xc, encut, ispin, nbands, npar, prec, nelm=80):
    lines = []
    xctags = {"pe": "PE", "pbe": "PBE",
              "ps": "PS", "pbesol": "PS",
              "ca": "CA", "lda": "CA"}
    lines.append("# common electronic part")
    lines.append("LREAL = .FALSE.")
    if xc is not None:
        xc = xctags.get(xc.lower(), None)
    if xc is not None:
        lines.append("GGA = %s" % xc)
    if encut is not None:
        lines.append("ENCUT = %f" % encut)
    if nbands is not None:
        lines.append("NBANDS = %d" % nbands)
    if npar is not None:
        lines.append("NPAR = %d" % npar)
    lines.append("ISPIN = %d" % ispin)
    lines.append("PREC = %s" % prec)
    lines.append("NELM = %s" % nelm)
    lines.append("ISMEAR = 0; SIGMA = 0.05 # fixed for semiconductor")
    return lines

def write_opt_incar(xc, encut, ispin, nbands, npar, prec, nelm=80,
                     force=0.01, nsw=30):
    """setup optimization input"""
    lines = _format_common_incar_lines(xc, encut, ispin, nbands, npar, prec, nelm)
    lines.append("")
    lines.append("# geometry optimization")
    lines.append("IBRION = 1")
    lines.append("ISIF = 3")
    lines.append("EDIFFG = -%f" % force)
    lines.append("NSW = %d" % nsw)
    return '\n'.join(lines)

def write_scf_incar(xc, encut, ispin, nbands, npar, prec, nelm=80):
    """setup optimization input"""
    lines = _format_common_incar_lines(xc, encut, ispin, nbands, npar, prec, nelm)
    #lines.append("ISMEAR = -5")
    return '\n'.join(lines)

def write_dfpt_incar(xc, encut, ispin, nbands, npar, prec):
    """setup DFPT input"""
    lines = _format_common_incar_lines(xc, encut, ispin, nbands, npar, prec)
    lines.append("")
    lines.append("# DFPT ")
    lines.append("LWAVE = .FALSE.")
    lines.append("LCHARG = .FALSE.")
    lines.append("ISTART = 1        # reuse SCF wfc")
    lines.append("LEPSILON = .TRUE. # compute dielectric tensor")
    lines.append("IBRION = 8        # phonon contribution")
    return '\n'.join(lines)

def write_emc_incar(xc, encut, ispin, nbands, npar, prec):
    """write input file for emc as well as band edge search"""
    lines = _format_common_incar_lines(xc, encut, ispin, nbands, npar, prec)
    lines.append("# nonSCF")
    lines.append("ICHARG = 11")
    lines.append("LWAVE = .FALSE.")
    lines.append("LCHARG = .FALSE.")
    return '\n'.join(lines)

def compute_emc(bandtag, nelect, ispin,
                kpt, stencil, stepsize, latt, Ns,
                mpi, nprocs, vasp):
    """compute the effective mass at the band edge at kpoint kpt

    if kpt is None, the band edge will be searched before emc calculation.

    Args:
        bandtag (str): 'vbm' or 'cbm'
        nelect (int)
        ispin (int)
        kpt (3-member list)
        stencil (int)
        stepsize (float)
        latt ((3,3) array)
        mpi (str)
        nprocs (int)
        vasp (str)
    """
    assert bandtag in ['vbm', 'cbm']
    ib = int(nelect/2)
    if bandtag == 'cbm':
        ib += 1
    if kpt is None:
        from scipy.optimize import brute
        def f(k):
            with open('KPOINTS', 'w') as h:
                h.write("search %s\n" % bandtag)
                h.write("1\nR\n%f %f %f 1.0\n" % (k[0], k[1], k[2]))
            run_vasp(mpi, nprocs, vasp)
            v = get_bande_from_eigenval(ib, ispin)[0]
            with open('searched_kpt.txt', 'a') as h:
                h.write("%f %f %f %f\n" % (k[0], k[1], k[2], v))
            # in VB we search the maximum
            if bandtag == 'vbm':
                v = -v
            return v
        # search in the whole BZ
        print("Searching %s..." % bandtag.upper())
        kpt = brute(f, ((0, 1), (0, 1), (0, 1)), Ns=Ns)
        search_note = "%s searched" % bandtag
    else:
        search_note = "%s specified" % bandtag
    with open('%s.dat' % bandtag, 'w') as h:
        print("# %s" % search_note, file=h)
        print("%s %s %s" % tuple(kpt), file=h)

    emc_setup(stencil, kpt, stepsize, ib, latt)
    run_vasp(mpi, nprocs, vasp)

def dfpt_analyze(outcar_fn="OUTCAR"):
    """analyze the DFPT output

    Namely, it retrieves the LO frequency and the static/ion-clamped dielectric constant.

    For dielectric constant, the trace average of the dielectric tensor is returned, i.e.

        eps = (eps_11 + eps_22 + eps_33) / 3
    """
    wLO = 0.0
    eps_inf = 1.0
    eps_s = 1.0
    with open(outcar_fn, 'r') as h:
        lines = h.readlines()
    for i, l in enumerate(lines):
        if l == " MACROSCOPIC STATIC DIELECTRIC TENSOR (including local field effects in DFT)\n":
            eps_inf = sum(map(float, [lines[i+2+j].split()[j] for j in range(3)])) / 3.0
        if l == " MACROSCOPIC STATIC DIELECTRIC TENSOR IONIC CONTRIBUTION\n":
            eps_s = eps_inf + sum(map(float, [lines[i+2+j].split()[j] for j in range(3)])) / 3.0
        if l == " Eigenvectors and eigenvalues of the dynamical matrix\n":
            wLO = float(lines[i+4].split()[3])
    return wLO, eps_inf, eps_s

def coupling_constant(effmass, wLO, eps_inf, eps_s):
    """compute the coupling constant in Frohlich model"""
    cc = (1.0/eps_inf - 1.0/eps_s) * np.sqrt(2.0*effmass/wLO/THZ2HA) / 2.0
    return cc

def renorm_e_frohlich(effmass, wLO, eps_inf, eps_s):
    """renormalization energy by frohlich model, in eV
    """
    cc = coupling_constant(effmass, wLO, eps_inf, eps_s)
    renorm_e = wLO * THZ2HA * HA2EV * (cc + 0.0159*cc**2 + 0.000806*cc**3)
    return renorm_e

def _parser():
    p = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    p.add_argument('-n', dest="nprocs", type=int, default=1,
                   help='<1> number of processors')
    p.add_argument('--mpi', type=str, default="mpirun",
                   help='<mpirun> mpi executable')
    p.add_argument('--opt', dest="flag_opt", action="store_true",
                   help="switch on geometry optimization")
    p.add_argument('--vasp', default='vasp_std', type=str,
                   help="<vasp_std> path to vasp executable")

    gv = p.add_argument_group("VASP arguments")
    gv.add_argument('-b', dest="nbands", default=None, type=int)
    gv.add_argument('-e', dest="encut", default=None, type=float)
    gv.add_argument('--xc', default=None, type=str,
                    choices=["pbe", "pe", "pbesol", "ps", "lda", "ca"],
                    help="xc functional (GGA). Default use PAW LEXCH")
    gv.add_argument('-k', dest="kpts", nargs=3, type=int, default=None,
                    help="kpoints (kx ky kz) for both SCF and DFPT, if no KPOINTS is provided")
    gv.add_argument('--prec', default="normal", choices=["normal", "N", "accurate", "A"], type=str)
    gv.add_argument('--npar', default=None, type=int)
    gv.add_argument('--ispin', default=1, choices=[1,], type=int)

    gs = p.add_argument_group("Band edge searching arguments")
    gs.add_argument('--kv', type=float, default=None, nargs=3,
                    help="kpoint of VBM. Default to None for automatic search")
    gs.add_argument('--kc', type=float, default=None, nargs=3,
                    help="kpoint of CBM. Default to None for automatic search")
    gs.add_argument('-N', type=int, dest="Ns", default=11,
                    help="<11> Ns parameter of scipy.optimize.brute")

    ge = p.add_argument_group("EMC arguments")
    ge.add_argument('--stc', dest="stencil", default=3, choices=[3, 5], type=int,
                    help="<3> stencil")
    ge.add_argument('-s', dest="stepsize", default=0.1, type=float,
                    help="<0.1> stepsize of finite difference")
    return p

def main():
    """the main stream"""
    args = _parser().parse_args()
    check_prereq(args.mpi, args.vasp, args.kpts)

    # Step 1
    if os.path.isdir('scf'):
        print('skip SCF: workdir scf found')
    else:
        os.mkdir('scf')
        os.chdir('scf')
        os.symlink('../POTCAR', 'POTCAR')
        copy2('../POSCAR', 'POSCAR')
        copy2('../KPOINTS', 'KPOINTS')
        if args.flag_opt:
            copy2('POSCAR', 'POSCAR.0')
            with open('INCAR.opt', 'w') as h:
                print(write_opt_incar(args.xc, args.encut, args.ispin,
                                      args.nbands, args.npar, args.prec), file=h)
            copy2('INCAR.opt', 'INCAR')
            run_vasp(args.mpi, args.nprocs, args.vasp)
            # TODO may need check opt convergence
            copy2('CONTCAR', 'POSCAR')
        with open('INCAR.scf', 'w') as h:
            print(write_scf_incar(args.xc, args.encut, args.ispin,
                                  args.nbands, args.npar, args.prec), file=h)
        copy2('INCAR.scf', 'INCAR')
        run_vasp(args.mpi, args.nprocs, args.vasp)
        os.chdir('..')

    # read the lattice
    latt = np.loadtxt('scf/POSCAR', skiprows=2, max_rows=3)
    with open('scf/POSCAR', 'r') as h:
        lines = h.readlines()
        latt = float(lines[1]) * latt

    # read electron number
    nelect = None
    with open('scf/OUTCAR', 'r') as h:
        l = h.readline()
        while l:
            m = match(r"   NELECT =\s*(\d+\.\d+)\s+", l)
            if m is not None:
                nelect = int(np.rint(float(m.group(1))))
                break
            l = h.readline()
    if nelect is None:
        raise IOError("Invalid OUTCAR in scf, please check")
    #print("NELECT = %d" % nelect)
    if nelect%2 == 1:
        print("Warning! Odd electron number, probably not a semiconductor")

    # Step 2: DFPT
    if os.path.isdir('dfpt'):
        print('skip DFPT: workdir dfpt found')
    else:
        os.mkdir('dfpt')
        os.chdir('dfpt')
        os.symlink('../POTCAR', 'POTCAR')
        copy2('../KPOINTS', 'KPOINTS')
        copy2('../scf/POSCAR', 'POSCAR')
        # reuse wfc in SCF
        if os.path.exists('../scf/WAVECAR'):
            copy2('../scf/WAVECAR', 'WAVECAR')
        with open('INCAR.dfpt', 'w') as h:
            print(write_dfpt_incar(args.xc, args.encut, args.ispin,
                                   args.nbands, args.npar, args.prec), file=h)
        copy2('INCAR.dfpt', 'INCAR')
        run_vasp(args.mpi, args.nprocs, args.vasp)
        os.chdir('..')

    # Step 3: compute effective mass at band edge
    # VBM
    if os.path.isdir('emc-vbm'):
        print('skip VBM EMC: workdir emc-vbm found')
    else:
        os.mkdir('emc-vbm')
        os.chdir('emc-vbm')
        os.symlink('../POTCAR', 'POTCAR')
        copy2('../scf/POSCAR', 'POSCAR')
        if not os.path.exists('../scf/CHGCAR'):
            raise IOError("broken SCF with no CHGCAR")
        copy2('../scf/CHGCAR', 'CHGCAR')
        with open('INCAR', 'w') as h:
            print(write_emc_incar(args.xc, args.encut, args.ispin,
                                  args.nbands, args.npar, args.prec), file=h)
        compute_emc('vbm', nelect, args.ispin,
                    args.kv, args.stencil, args.stepsize, latt, args.Ns,
                    args.mpi, args.nprocs, args.vasp)
        os.chdir('..')
    # CBM
    if os.path.isdir('emc-cbm'):
        print('skip CBM EMC: workdir emc-cbm found')
    else:
        os.mkdir('emc-cbm')
        os.chdir('emc-cbm')
        os.symlink('../POTCAR', 'POTCAR')
        copy2('../scf/POSCAR', 'POSCAR')
        if not os.path.exists('../scf/CHGCAR'):
            raise IOError("broken SCF with no CHGCAR")
        copy2('../scf/CHGCAR', 'CHGCAR')
        with open('INCAR', 'w') as h:
            print(write_emc_incar(args.xc, args.encut, args.ispin,
                                  args.nbands, args.npar, args.prec), file=h)
        compute_emc('cbm', nelect, args.ispin,
                    args.kc, args.stencil, args.stepsize, latt, args.Ns,
                    args.mpi, args.nprocs, args.vasp)
        os.chdir('..')

    # after all these steps, start analysis
    os.chdir('dfpt')
    wLO, eps_inf, eps_s = dfpt_analyze()
    print("DFPT results:")
    print("-------------")
    print("        w_LO (THz) = %f" % wLO)
    print("  epsilon (elect.) = %f" % eps_inf)
    print("  epsilon (static) = %f" % eps_s)
    print("")

    print("Effective mass and EPC correction:")
    print("----------------------------------")

    os.chdir('../emc-vbm')
    mass= emc_analyze(args.stencil, args.stepsize, int(nelect/2), latt)
    kv = np.loadtxt('vbm.dat')
    print("  VBM: @(%12.6f, %12.6f, %12.6f)" % (kv[0], kv[1], kv[2]))
    print("       eff. mass: (%10.4f, %10.4f, %10.4f)" % (mass[0], mass[1], mass[2]))
    geoavg = np.power(np.absolute(np.product(mass)), 1.0/3.0)
    print("       geo. aveg: %10.4f" % geoavg)
    print("       coup. cst: %10.4f" % coupling_constant(geoavg, wLO, eps_inf, eps_s))
    print("     renorm (eV): +%9.4f" % renorm_e_frohlich(geoavg, wLO, eps_inf, eps_s))

    os.chdir('../emc-cbm')
    mass = emc_analyze(args.stencil, args.stepsize, int(nelect/2)+1, latt)
    kc = np.loadtxt('cbm.dat')
    print("  CBM: @(%12.6f, %12.6f, %12.6f)" % (kc[0], kc[1], kc[2]))
    print("       eff. mass: (%10.4f, %10.4f, %10.4f)" % (mass[0], mass[1], mass[2]))
    geoavg = np.power(np.absolute(np.product(mass)), 1.0/3.0)
    print("       geo. aveg: %10.4f" % geoavg)
    print("       coup. cst: %10.4f" % coupling_constant(geoavg, wLO, eps_inf, eps_s))
    print("     renorm (eV): -%9.4f" % renorm_e_frohlich(geoavg, wLO, eps_inf, eps_s))

    os.chdir('..')

if __name__ == "__main__":
    main()

