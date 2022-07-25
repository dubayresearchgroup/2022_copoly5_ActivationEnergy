import sys
import itertools
import math
import numpy as np
import atom

def grids(xrange, yrange, zrange):
    """Creates an Numpy array that contains all integer combinations of [xrange, yrange, zrange] for the range of integer values provided by range(xrange/yrange/zrange).
    
    Parameters
    ----------
    xrange : int
        Range of integer values in the x position
    yrange : int
        Range of integer values in the y position
    zrange : int
        Range of integer values in the z position
    """
    for x in range(xrange):
        for y in range(yrange):
            for z in range(zrange):
                yield np.array([x, y, z])


def writefile(filename, size, numA, numB, shiftA=0, shiftB=0):
    """Writes LAMMPS data file containing 3 atom types, 3 bond types, and 3 angle types, with specified box size, number of monomers, and specified monomer sizes.

    Parameters
    ----------
    filename : str
        Name of file to write LAMMPS data file to.
    size : float
        Size of one side of simulation box, here all dimensions of simulation box have equal dimensions.
    numA : int
        Number of monomers of type A
    numB : int
        Number of monomers of type B
    shiftA : float
        The radius of monomer A is calculated from this value by placing the peripheral atoms at shiftA+0.4 and setting the bond length betweeen these atoms and the center atom to this value.
    shiftB : float
        The radius of monomer B is calculated from this value by placing the peripheral atoms at shiftA+0.4 and setting the bond length betweeen these atoms and the center atom to this value.
    """
    numA = int(numA)
    numB = int(numB)
    data = atom.Data()
    data.setheader(size, 3, 3, 3)
    monosize = 1+max(shiftA,shiftB)
    if (size // monosize)**3 < numA + numB:
        raise Exception('not enough space for monomers in this method.')
    centertype = {'A':2, 'B':3}
    angletype = {'A':2, 'B':3}
    bondtype = {'A':2, 'B':3}
    shift = {'A':shiftA, 'B':shiftB}
    pool = ['A']*numA + ['B']*numB
    disp = np.array([monosize/2, monosize/2, monosize/2])
    bin = math.ceil((numA+numB)**(1/3))
    interval = size / bin
    gridpoint = grids(bin, bin, bin)
    for monomer in pool:
        pos = next(gridpoint)*interval + disp
        data.monomers.newmono(pos, centertype[monomer], bondtype[monomer], angletype[monomer], shift[monomer])
    with open(filename, 'w') as fp:
        data.writefile(fp)

helptxt = \
'''
usage: mono_prep.py options... filename

options:
-numA=xx, -numB=xx        number of A and B monomers
-shiftA=xx, -shiftB=xx    shift of monomers (increment in diameter)
-size=xx                  simulation box size

'''

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(helptxt)
        exit()
    kws = {}
    for arg in sys.argv[1:]:
        if arg.startswith('-'):
            for kw in ['size', 'numA', 'numB', 'shiftA', 'shiftB']:
                if arg.startswith(f'-{kw}='):
                    kws[kw] = float(arg.lstrip(f'-{kw}='))

        else:
            kws['filename'] = arg
            for kw in ['size', 'numA', 'numB']:
                if kw not in kws:
                    print(f'error: argument {kw} not specified.')
                    kws = {}
                    continue
            writefile(**kws)
            kws = {}
    
