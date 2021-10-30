#!/usr/bin/env python
# pylint: disable=W0301,C0301,C0321,C0200,C0116
"""effective mass calculator, originally written by Alexandr Fonari and Chris Sutton

Orignal file: https://github.com/afonari/emc/blob/master/emc.py
License: MIT, see https://github.com/afonari/emc/blob/master/LICENSE
"""
from __future__ import print_function
import sys
import time

from numpy import zeros
from numpy.linalg import eigh, norm

EMC_VERSION = '1.51py'
#
###################################################################################################
#
#   STENCILS for finite difference
def set_stencil_fd(stencil):
    if stencil == 3:
        return _stencil_fd_3()
    if stencil == 5:
        return _stencil_fd_5()
    raise ValueError("stencil should be either 3 or 5")
#
#   three-point stencil
def _stencil_fd_3():
    st3 = []
    st3.append([0.0, 0.0, 0.0]); # 0
    st3.append([-1.0, 0.0, 0.0]);  st3.append([1.0, 0.0, 0.0]);  # dx  1-2
    st3.append([0.0, -1.0, 0.0]);  st3.append([0.0, 1.0, 0.0])   # dy  3-4
    st3.append([0.0, 0.0, -1.0]);  st3.append([0.0, 0.0, 1.0])   # dz  5-6
    st3.append([-1.0, -1.0, 0.0]); st3.append([1.0, 1.0, 0.0]); st3.append([1.0, -1.0, 0.0]); st3.append([-1.0, 1.0, 0.0]); # dxdy 7-10
    st3.append([-1.0, 0.0, -1.0]); st3.append([1.0, 0.0, 1.0]); st3.append([1.0, 0.0, -1.0]); st3.append([-1.0, 0.0, 1.0]); # dxdz 11-14
    st3.append([0.0, -1.0, -1.0]); st3.append([0.0, 1.0, 1.0]); st3.append([0.0, 1.0, -1.0]); st3.append([0.0, -1.0, 1.0]); # dydz 15-18
    def fd_effmass_st3(e, h):
        m = [[0.0 for i in range(3)] for j in range(3)]
        m[0][0] = (e[1] - 2.0*e[0] + e[2])/h**2
        m[1][1] = (e[3] - 2.0*e[0] + e[4])/h**2
        m[2][2] = (e[5] - 2.0*e[0] + e[6])/h**2

        m[0][1] = (e[7] + e[8] - e[9] - e[10])/(4.0*h**2)
        m[0][2] = (e[11] + e[12] - e[13] - e[14])/(4.0*h**2)
        m[1][2] = (e[15] + e[16] - e[17] - e[18])/(4.0*h**2)

        # symmetrize
        m[1][0] = m[0][1]
        m[2][0] = m[0][2]
        m[2][1] = m[1][2]
        ##
        #print('-> fd_effmass_st3: Effective mass tensor:\n')
        #for i in range(len(m)):
        #    print('%15.8f %15.8f %15.8f' % (m[i][0], m[i][1], m[i][2]))
        #print('')
        ##
        return m
    return st3, fd_effmass_st3
#
#   five-point stencil
def _stencil_fd_5():
    st5 = []
    st5.append([0.0, 0.0, 0.0])
    #
    a = [-2,-1,1,2]
    for i in range(len(a)): #dx
        st5.append([float(a[i]), 0., 0.])
    #
    for i in range(len(a)): #dy
        st5.append([0., float(a[i]), 0.])
    #
    for i in range(len(a)): #dz
        st5.append([0., 0., float(a[i])])
    #
    for i in range(len(a)):
        i1=float(a[i])
        for j in range(len(a)):
            j1=float(a[j])
            st5.append([j1, i1, 0.]) # dxdy
    #
    for i in range(len(a)):
        i1=float(a[i])
        for j in range(len(a)):
            j1=float(a[j])
            st5.append([j1, 0., i1,]) # dxdz
    #
    for i in range(len(a)):
        i1=float(a[i])
        for j in range(len(a)):
            j1=float(a[j])
            st5.append([0., j1, i1]) # dydz

    def fd_effmass_st5(e, h):
        m = [[0.0 for i in range(3)] for j in range(3)]
        #
        m[0][0] = (-(e[1]+e[4])  + 16.0*(e[2]+e[3])   - 30.0*e[0])/(12.0*h**2)
        m[1][1] = (-(e[5]+e[8])  + 16.0*(e[6]+e[7])   - 30.0*e[0])/(12.0*h**2)
        m[2][2] = (-(e[9]+e[12]) + 16.0*(e[10]+e[11]) - 30.0*e[0])/(12.0*h**2)
        #
        m[0][1] = (-63.0*(e[15]+e[20]+e[21]+e[26]) + 63.0*(e[14]+e[17]+e[27]+e[24]) \
                   +44.0*(e[16]+e[25]-e[13]-e[28]) + 74.0*(e[18]+e[23]-e[19]-e[22]))/(600.0*h**2)
        m[0][2] = (-63.0*(e[31]+e[36]+e[37]+e[42]) + 63.0*(e[30]+e[33]+e[43]+e[40]) \
                   +44.0*(e[32]+e[41]-e[29]-e[44]) + 74.0*(e[34]+e[39]-e[35]-e[38]))/(600.0*h**2)
        m[1][2] = (-63.0*(e[47]+e[52]+e[53]+e[58]) + 63.0*(e[46]+e[49]+e[59]+e[56]) \
                   +44.0*(e[48]+e[57]-e[45]-e[60]) + 74.0*(e[50]+e[55]-e[51]-e[54]))/(600.0*h**2)
        #
        # symmetrize
        m[1][0] = m[0][1]
        m[2][0] = m[0][2]
        m[2][1] = m[1][2]
        ##
        #print('-> fd_effmass_st5: Effective mass tensor:\n')
        #for i in range(3):
        #    print('%15.8f %15.8f %15.8f' % (m[i][0], m[i][1], m[i][2]))
        #print('')
        ##
        return m
    return st5, fd_effmass_st5

#
#   CONSTANTS
#
BOHR2ANG = 0.52917721092
EV2H = 1.0/27.21138505
#
#######  FUNCTIONS and __main__  ##################################################################
#
def MAT_m_VEC(m, v):
    p = [ 0.0 for i in range(len(v)) ]
    for i in range(len(m)):
        assert len(v) == len(m[i]), 'Length of the matrix row is not equal to the length of the vector'
        p[i] = sum( [ m[i][j]*v[j] for j in range(len(v)) ] )
    return p

def T(m):
    p = [[ m[i][j] for i in range(len( m[j] )) ] for j in range(len( m )) ]
    return p

def N(v):
    """normalize"""
    max_ = 0.
    for item in v:
        if abs(item) > abs(max_): max_ = item

    return [ item/max_ for item in v ]

def DET_3X3(m):
    assert len(m) == 3, 'Matrix should be of the size 3 by 3'
    return m[0][0]*m[1][1]*m[2][2] + m[1][0]*m[2][1]*m[0][2] + m[2][0]*m[0][1]*m[1][2] - \
           m[0][2]*m[1][1]*m[2][0] - m[2][1]*m[1][2]*m[0][0] - m[2][2]*m[0][1]*m[1][0]

def SCALE_ADJOINT_3X3(m, s):
    a = [[0.0 for i in range(3)] for j in range(3)]

    a[0][0] = (s) * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
    a[1][0] = (s) * (m[1][2] * m[2][0] - m[1][0] * m[2][2])
    a[2][0] = (s) * (m[1][0] * m[2][1] - m[1][1] * m[2][0])

    a[0][1] = (s) * (m[0][2] * m[2][1] - m[0][1] * m[2][2])
    a[1][1] = (s) * (m[0][0] * m[2][2] - m[0][2] * m[2][0])
    a[2][1] = (s) * (m[0][1] * m[2][0] - m[0][0] * m[2][1])

    a[0][2] = (s) * (m[0][1] * m[1][2] - m[0][2] * m[1][1])
    a[1][2] = (s) * (m[0][2] * m[1][0] - m[0][0] * m[1][2])
    a[2][2] = (s) * (m[0][0] * m[1][1] - m[0][1] * m[1][0])

    return a

def INVERT_3X3(m):
    tmp = 1.0/DET_3X3(m)
    return SCALE_ADJOINT_3X3(m, tmp)

def IS_SYMMETRIC(m):
    for i in range(len(m)):
        for j in range(len(m[i])):
            if m[i][j] != m[j][i]: return False # automatically checks square-shape

    return True

#def jacobi(ainput):
#    # from NWChem/contrib/python/mathutil.py
#    # possible need to rewrite due to licensing issues
#    #
#    from math import sqrt
#    #
#    a = [[ ainput[i][j] for i in range(len( ainput[j] )) ] for j in range(len( ainput )) ] # copymatrix
#    n = len(a)
#    m = len(a[0])
#    if n != m:
#        raise 'jacobi: Matrix must be square'
#    #
#    for i in range(n):
#        for j in range(m):
#            if a[i][j] != a[j][i]:
#                raise 'jacobi: Matrix must be symmetric'
#    #
#    tolmin = 1e-14
#    tol = 1e-4
#    #
#    v = [[0.0 for i in range(n)] for j in range(n)] # zeromatrix
#    for i in range(n):
#        v[i][i] = 1.0
#    #
#    maxd = 0.0
#    for i in range(n):
#        maxd = max(abs(a[i][i]),maxd)
#    #
#    for iter in range(50):
#        nrot = 0
#        for i in range(n):
#            for j in range(i+1,n):
#                aii = a[i][i]
#                ajj = a[j][j]
#                daij = abs(a[i][j])
#                if daij > tol*maxd: # Screen small elements
#                    nrot = nrot + 1
#                    s = aii - ajj
#                    ds = abs(s)
#                    if daij > (tolmin*ds): # Check for sufficient precision
#                        if (tol*daij) > ds:
#                            c = s = 1/sqrt(2.)
#                        else:
#                            t = a[i][j]/s
#                            u = 0.25/sqrt(0.25+t*t)
#                            c = sqrt(0.5+u)
#                            s = 2.*t*u/c
#                        #
#                        for k in range(n):
#                            u = a[i][k]
#                            t = a[j][k]
#                            a[i][k] = s*t + c*u
#                            a[j][k] = c*t - s*u
#                        #
#                        for k in range(n):
#                            u = a[k][i]
#                            t = a[k][j]
#                            a[k][i] = s*t + c*u
#                            a[k][j]= c*t - s*u
#                        #
#                        for k in range(n):
#                            u = v[i][k]
#                            t = v[j][k]
#                            v[i][k] = s*t + c*u
#                            v[j][k] = c*t - s*u
#                        #
#                        a[j][i] = a[i][j] = 0.0
#                        maxd = max(maxd,abs(a[i][i]),abs(a[j][j]))
#        #
#        if nrot == 0 and tol <= tolmin:
#            break
#        tol = max(tolmin,tol*0.99e-2)
#    #
#    if nrot != 0:
#        print('jacobi: [WARNING] Jacobi iteration did not converge in 50 passes!')
#    #
#    # Sort eigenvectors and values into increasing order
#    e = [0.0 for i in range(n)] # zerovector
#    for i in range(n):
#        e[i] = a[i][i]
#        for j in range(i):
#            if e[j] > e[i]:
#                (e[i],e[j]) = (e[j],e[i])
#                (v[i],v[j]) = (v[j],v[i])
#    #
#    return (v,e)
#
def cart2frac(basis, v):
    return MAT_m_VEC( T(INVERT_3X3(basis)), v )

def generate_kpoints(kpt_frac, st, h, basis):
    from math import pi
    #
    # working in the reciprocal space
    m = INVERT_3X3(T(basis))
    basis_r = [[ m[i][j]*2.0*pi for j in range(3) ] for i in range(3) ]
    #
    kpt_rec = MAT_m_VEC(T(basis_r), kpt_frac)
    #print('-> generate_kpoints: K-point in reciprocal coordinates: %5.3f %5.3f %5.3f' % (kpt_rec[0], kpt_rec[1], kpt_rec[2]))
    #
    #if prg == 'V' or prg == 'P':
    #    h = h*(1/BOHR2ANG) # [1/A]
    h = h*(1/BOHR2ANG) # [1/A]
    #
    kpoints = []
    for i in range(len(st)):
        k_c_ = [ kpt_rec[j] + st[i][j]*h for j in range(3) ] # getting displaced k points in Cartesian coordinates
        k_f = cart2frac(basis_r, k_c_)
        kpoints.append( [k_f[0], k_f[1], k_f[2]] )
    #
    return kpoints

#def parse_bands_CASTEP(eigenval_fh, band, diff2_size, debug=False):
#
#    # Number of k-points X
#    nkpt = int(eigenval_fh.readline().strip().split()[3])
#
#    # Number of spin components X
#    spin_components = float(eigenval_fh.readline().strip().split()[4])
#
#    # Number of electrons X.00 Y.00
#    tmp = eigenval_fh.readline().strip().split()
#    if spin_components == 1:
#        nelec = int(float(tmp[3]))
#        n_electrons_down = None
#    elif spin_components == 2:
#        nelec = [float(tmp[3])]
#        n_electrons_down = int(float(tmp[4]))
#
#    # Number of eigenvalues X
#    nband = int(eigenval_fh.readline().strip().split()[3])
#
#    energies = []
#    # Get eigenenergies and unit cell from .bands file
#    while True:
#        line = eigenval_fh.readline()
#        if not line:
#            break
#        #
#        if 'Spin component 1' in line:
#            for i in range(1, nband + 1):
#                energy = float(eigenval_fh.readline().strip())
#                if band == i:
#                    energies.append(energy)
#
#    return energies

def parse_EIGENVAL_VASP(eigenval_fh, band, diff2_size, debug=False):
    eigenval_fh.seek(0) # just in case
    eigenval_fh.readline()
    eigenval_fh.readline()
    eigenval_fh.readline()
    eigenval_fh.readline()
    eigenval_fh.readline()
    #
    nelec, nkpt, nband = [int(s) for s in eigenval_fh.readline().split()]
    #if debug: print('From EIGENVAL: Number of the valence band is %d (NELECT/2)' % (nelec/2))
    if band > nband:
        raise ValueError('Requested band (%d) is larger than total number of the calculated bands (%d)!' % (band, nband))

    energies = []
    for i in range(diff2_size):
        eigenval_fh.readline() # empty line
        eigenval_fh.readline() # k point coordinates
        for j in range(1, nband+1):
            line = eigenval_fh.readline()
            if band == j:
                energies.append(float(line.split()[1])*EV2H)

    #if debug: print('')
    return energies
#
#def parse_nscf_PWSCF(eigenval_fh, band, diff2_size, debug=False):
#    eigenval_fh.seek(0) # just in case
#    engrs_at_k = []
#    energies = []
#    #
#    while True:
#        line = eigenval_fh.readline()
#        if not line:
#            break
#        #
#        if "End of band structure calculation" in line:
#            for i in range(diff2_size):
#                #
#                while True:
#                    line = eigenval_fh.readline()
#                    if "occupation numbers" in line:
#                        break
#                    #
#                    if "k =" in line:
#                        a = [] # energies at a k-point
#                        eigenval_fh.readline() # empty line
#                        #
#                        while True:
#                            line = eigenval_fh.readline()
#                            if line.strip() == "": # empty line
#                                break
#                            #
#                            a.extend(line.strip().split())
#                        #
#                        #print(a)
#                        assert len(a) <= band, 'Length of the energies array at a k-point is smaller than band param'
#                        energies.append(float(a[band-1])*EV2H)
#    #
#    #print(engrs_at_k)
#    return energies
##
#def parse_inpcar(inpcar_fh, debug=False):
#    import sys
#    import re
#    #
#    kpt = []       # k-point at which eff. mass in reciprocal reduced coords (3 floats)
#    stepsize = 0.0 # stepsize for finite difference (1 float) in Bohr
#    band = 0       # band for which eff. mass is computed (1 int)
#    prg = ''       # program identifier (1 char)
#    basis = []     # basis vectors in cartesian coords (3x3 floats), units depend on the program identifier
#    #
#    inpcar_fh.seek(0) # just in case
#    p = re.search(r'^\s*(-*\d+\.\d+)\s+(-*\d+\.\d+)\s+(-*\d+\.\d+)', inpcar_fh.readline())
#    if p:
#        kpt = [float(p.group(1)), float(p.group(2)), float(p.group(3))]
#        if debug: print("Found k point in the reduced reciprocal space: %5.3f %5.3f %5.3f" % (kpt[0], kpt[1], kpt[2]))
#    else:
#        print("Was expecting k point on the line 0 (3 floats), didn't get it, exiting...")
#        sys.exit(1)
#
#    p = re.search(r'^\s*(\d+\.\d+)', inpcar_fh.readline())
#    if p:
#        stepsize = float(p.group(1))
#        if debug: print("Found stepsize of: %5.3f (1/Bohr)" % stepsize)
#    else:
#        print("Was expecting a stepsize on line 1 (1 float), didn't get it, exiting...")
#        sys.exit(1)
#
#    p = re.search(r'^\s*(\d+)', inpcar_fh.readline())
#    if p:
#        band = int(p.group(1))
#        if debug: print("Requested band is : %5d" % band)
#    else:
#        print("Was expecting band number on line 2 (1 int), didn't get it, exiting...")
#        sys.exit(1)
#
#    p = re.search(r'^\s*(\w)', inpcar_fh.readline())
#    if p:
#        prg = p.group(1)
#        if debug: print("Program identifier is: %5c" % prg)
#    else:
#        print("Was expecting program identifier on line 3 (1 char), didn't get it, exiting...")
#        sys.exit(1)
#
#    for i in range(3):
#        p = re.search(r'^\s*(-*\d+\.\d+)\s+(-*\d+\.\d+)\s+(-*\d+\.\d+)', inpcar_fh.readline())
#        if p:
#            basis.append([float(p.group(1)), float(p.group(2)), float(p.group(3))])
#
#    if debug:
#        print("Real space basis:")
#        for i in range(len(basis)):
#            print('%9.7f %9.7f %9.7f' % (basis[i][0], basis[i][1], basis[i][2]))
#
#    if debug: print('')
#
#    return kpt, stepsize, band, prg, basis

def get_eff_masses(m, basis):
    #
    #vecs_cart = [[0.0 for i in range(3)] for j in range(3)]
    #vecs_frac = [[0.0 for i in range(3)] for j in range(3)]
    #vecs_n    = [[0.0 for i in range(3)] for j in range(3)]
    vecs_cart = zeros((3, 3), dtype='float64')
    vecs_frac = zeros((3, 3), dtype='float64')
    vecs_n    = zeros((3, 3), dtype='float64')
    ## original
    #eigvec, eigval = jacobi(m)
    # using numpy function
    eigval, eigvec = eigh(m)
    # NOTE: the eigenvalues are computed the same as the original code
    #       the eigenvectors may slightly differ when there are degenerate directions
    # TEST: C diamond
    for i in range(3):
        vecs_cart[i, :] = eigvec[:, i]
        vecs_frac[i, :] = cart2frac(basis, eigvec[:, i])
        vecs_n[i, :]    = N(vecs_frac[i])
        #vecs_n[i, :]    = vecs_frac[i, :] / norm(vecs_frac[i, :])
    #
    em = 1.0 / eigval
    return em, vecs_cart, vecs_frac, vecs_n

def emc_header(stencil):
    import datetime
    lines = []
    lines.append('Effective mass calculator '+EMC_VERSION)
    lines.append('Stencil: '+str(stencil))
    lines.append('License: MIT')
    lines.append('Developed by: Alexandr Fonari and Chris Sutton')
    lines.append('Started at: '+datetime.datetime.now().strftime("%Y-%m-%d %H:%M")+'\n')
    return '\n'.join(lines)

#
def emc_setup(stencil, kpt, stepsize, band, basis):
    filename = 'emcpy.out_'+str(int(time.time()))
    print('Redirecting output to '+filename)
    with open(filename, 'w') as fh:
        print(emc_header(stencil), file=fh)
        # set up the stencil
        st, fd_effmass = set_stencil_fd(stencil)
        # set up KPOINTS file
        kpoints = generate_kpoints(kpt, st, stepsize, basis)
        with open('KPOINTS', 'w') as kpoints_fh:
            kpoints_fh.write("EMC "+EMC_VERSION+"\n")
            kpoints_fh.write("%d\n" % len(st))
            kpoints_fh.write("Reciprocal\n")
            for k in kpoints:
                kpoints_fh.write( '%15.10f %15.10f %15.10f 0.01\n' % (k[0], k[1], k[2]) )
        print('KPOINTS file has been generated in the current directory...', file=fh)

    # create emc.in file, such that the original emc.py can read and the results can be compared
    with open('emc.in', 'w') as h:
        h.write("%f %f %f\n" % (kpt[0], kpt[1], kpt[2]))
        h.write("%f\n" % stepsize)
        h.write("%d\n" % band)
        h.write("V\n")
        for i in range(3):
            h.write("%f %f %f\n" % (basis[i][0], basis[i][1], basis[i][2]))

def emc_analyze(stencil, stepsize, band, basis, output_fn='EIGENVAL'):
    filename = 'emcpy.out_'+str(int(time.time()))
    print('Redirecting output to '+filename)
    with open(filename, 'w') as fh:
        print(emc_header(stencil), file=fh)
        # set up the stencil
        st, fd_effmass = set_stencil_fd(stencil)
        with open(output_fn, 'r') as output_fh:
            energies = []
            energies = parse_EIGENVAL_VASP(output_fh, band, len(st))
            m = fd_effmass(energies, stepsize)
            print('Effective mass tensor:\n', file=fh)
            for i in range(3):
                print('%15.8f %15.8f %15.8f' % (m[i][0], m[i][1], m[i][2]), file=fh)
            print('', file=fh)
            #if prg.upper() == 'V' or prg.upper() == 'C':
            #    energies = parse_EIGENVAL_VASP(output_fh, band, len(st))
            #    m = fd_effmass(energies, stepsize)
            ##
            #if prg.upper() == 'Q':
            #    energies = parse_nscf_PWSCF(output_fh, band, len(st))
            #    m = fd_effmass(energies, stepsize)
            ##
            #if prg.upper() == 'P':
            #    energies = parse_bands_CASTEP(output_fh, band, len(st))
            #    m = fd_effmass(energies, stepsize)
            ##
            masses, vecs_cart, vecs_frac, vecs_n = get_eff_masses(m, basis)
            print('Principle effective masses and directions:\n', file=fh)
            for i in range(3):
                print('Effective mass (%d): %12.3f' % (i, masses[i]), file=fh)
                print('Original eigenvectors: %7.5f %7.5f %7.5f' % (vecs_cart[i][0], vecs_cart[i][1], vecs_cart[i][2]), file=fh)
                print('Normal fractional coordinates: %7.5f %7.5f %7.5f\n' % (vecs_n[i][0], vecs_n[i][1], vecs_n[i][2]), file=fh)
    return masses

