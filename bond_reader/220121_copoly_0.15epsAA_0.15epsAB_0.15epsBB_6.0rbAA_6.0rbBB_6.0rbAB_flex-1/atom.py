import itertools
import collections
import numpy as np

class Atoms:
    """Class encapsulating a set of Atom data structures containing an Atom ID, Molecule ID, Atom Type, and the (x,y,z) coordinates.
    The atom ID is automatically calculated using itertools.count starting at 1.
    """
    Atom = collections.namedtuple('Atom', ['mid', 'atype', 'pos'])
    def __init__(self):
        self.counter = itertools.count(start=1)
        self.data = {}

    def newatom(self, mid, atype, pos):
        """Adds a new atom to the Atoms class instance with specified molecule id, atom type, and position.
        
        Parameters
        ----------
        mid : int
            The molecule id of the new atom to be added.
        atype : int
            The atom type of the new atom to be added.
        pos : list of float
            A list of floats containing the x, y, and z coordinates of the new atom to be added.

        Returns
        -------
        ID : int
            The automatically generated atom ID of the new atom added to the Atoms class instance.
        
        """
        ID = next(self.counter)
        self.data[ID] = self.Atom(mid, atype, pos)
        return ID
    
    def count(self):
        """
        Gives the number of atoms contained within the Atoms class instance.

        Returns
        -------
        int
            The number of atoms currently contained in the Atoms class instance.
        """
        return len(self.data)

    def section(self):
        """Returns the atoms in the Atoms class instance as a string formatted for a LAMMPS data file Atoms section.

        Returns
        -------
        str
            A string formatted for a LAMMPS Atoms section containing all atoms within the Atoms class instance.
        """
        out = 'Atoms\n\n'
        for ID in self.data:
            atom = self.data[ID]
            pos = atom.pos
            out += f'{ID} {atom.mid} {atom.atype} {pos[0]:g} {pos[1]:g} {pos[2]:g}\n'
        out += '\n'
        return out
    
class Bonds:
    """A class encapsulating a set of bond data structures containing the bond ID, bond type, and bonding atom IDs.
    The bond ID is automatically generated via an itertools counter starting at 1.
    """
    Bond = collections.namedtuple('Bond', ['btype', 'atoms'])
    def __init__(self):
        self.data = {}
        self.counter = itertools.count(start=1)

    def newbond(self, btype, atoms):
        """Adds a new bond to the Bonds class instance with the specified bond type and bonding atom IDs.

        Parameters
        ----------
        btype : int
            The bond type of the new bond to be added to the Bonds class instance.
        atoms : list of int
            A list of the atom IDs of the atoms bonded with this bond.

        Returns
        -------
        ID : int
            The generated bond ID of the newly added bond.
        """
        ID = next(self.counter)
        self.data[ID] = self.Bond(btype, atoms)
        return ID

    def count(self):
        """Returns the number of bonds in theis Bonds class instance.

        Returns
        -------
        int
            The number of bonds currently in the Bonds class instance.
        """
        return len(self.data)

    def section(self):
        """Returns a string formatted for a LAMMPS data file Bonds section.

        Returns
        -------
        str
            A string containing the Bonds section of a LAMMPS data file that contains all bonds currently in the Bonds class instance.
        """
        out = 'Bonds\n\n'
        for ID in self.data:
            bond = self.data[ID]
            atoms = bond.atoms
            out += f'{ID} {bond.btype} {atoms[0]} {atoms[1]}\n'
        out += '\n'
        return out


class Angles:
    """A class encapsulating a set of angle data structures each containing the angle id, angle type, and the atom IDs of the atoms in the angle.
    The angle ID is automatically generated starting at 1.
    """
    Angle = collections.namedtuple('Angle', ['atype', 'atoms'])
    def __init__(self):
        self.data = {}
        self.counter = itertools.count(start=1)

    def newangle(self, atype, atoms):
        """Adds a new angle to the Angles class instance.
        
        Parameters
        ----------
        atype : int
            The angle type of the angle to be added.
        atoms : list of int
            The atom IDs of the atoms contained in the angle to be added

        Returns
        -------
        ID : int
            The angle ID of the new angle added.
        """
        ID = next(self.counter)
        self.data[ID] = self.Angle(atype, atoms)
        return ID

    def count(self):
        """Returns the number of angles in the Angles class instance.

        Returns
        -------
        int
            The number of angles contained within the Angles class instance.
        """
        return len(self.data)

    def section(self):
        """Returns a string formatted for a LAMMPS data file Angles section
        
        Returns
        -------
        str
            A string formatted for a LAMMPS data file Angles section.
        """
        out = 'Angles\n\n'
        for ID in self.data:
            angle = self.data[ID]
            atoms = angle.atoms
            out += f'{ID} {angle.atype} {atoms[0]} {atoms[1]} {atoms[2]}\n'
        out += '\n'
        return out

class Monomers:
    """A class encapsulating a set of monomer data structures each containing the atoms, bonds, and angles that belong to each monomer.
    Note that this class is currently only equipped for monomers with only three atoms.
    """
    Monomer = collections.namedtuple('Monomer', ['atoms', 'bonds', 'angles'])
    def __init__(self, atoms, bonds, angles):
        self.atoms = atoms
        self.bonds = bonds
        self.angles = angles

        self.atom_link_type = 1
        self.angle_link_type = 1

    def newmono(self, center, atomtype, bondtype, angletype, shift=0):
        """Adds a new monomer to the Monomers class instance with specified position, atom type, bond type, angle type.

        Parameters
        ----------
        center : list of float
            Centered [x, y, z] position of the monomer
        atomtype : int
            The atom type of the middle atom in the three atom monomer
        bondtype : int
            Bond type of the bonds connecting the three monomer atoms.
        angletype : int
            The angle type of the angle connecting the three monomers together.
        shift : float
            The shift in position from the center atom of the two peripheral atoms, an initial shift of 0.4 is added onto this shift to prevent placing the peripheral atoms within the VdW radius of the center atom.  
            This shift is also used in prep_mono.py to change monomer diameter by also setting the bond length between the peripheral atoms and the center atom to this length plus 0.4

        Returns
        -------
        mono : Monomer object
            A Monomer object containing the newly created Monomer for the Monomers class instance.  Monomer is a named tuple collection class inside the Monomers class.
        """
        mono = Monomers([], [], [])

        mono.atoms.append(self.atoms.newatom(0, self.atom_link_type, center+np.array([0, 0, -0.4-shift/2])))
        mono.atoms.append(self.atoms.newatom(0, atomtype, center))
        mono.atoms.append(self.atoms.newatom(0, self.atom_link_type, center+np.array([0, 0, 0.4+shift/2])))

        mono.bonds.append(self.bonds.newbond(bondtype, mono.atoms[0:2]))
        mono.bonds.append(self.bonds.newbond(bondtype, mono.atoms[1:3]))

        mono.angles.append(self.angles.newangle(angletype, mono.atoms))

        return mono


class Data:
    """A class encapsulating a LAMMPS data file.  Used for writing a set of Atoms, Bonds, Angles to a LAMMPS data file 
for simulation and includes specification for 2 extra bonds per atom, 1 extra angle per atom, and 12 extra specials per atom.
    """
    def __init__(self):
        self.atoms = Atoms()
        self.bonds = Bonds()
        self.angles = Angles()
        self.monomers = Monomers(self.atoms, self.bonds, self.angles)

    Headerparam = collections.namedtuple('Headerparam', ['size', 'atomtypes', 'bondtypes', 'angletypes'])
    def setheader(self, size, atomtypes, bondtypes, angletypes):
        """Sets the LAMMPS data file header details.
        
        Parameters
        ----------
        size : float
            Length of a side of the simulation box.  Box is a cube by default so the length specified here is used for all dimensions.
        atomtypes : int
            Number of atom types used in the LAMMPS data file.
        bondtypes : int
            Number of bond types used in the LAMMPS data file.
        angletypes : int
            Number of angle types used in the LAMMPS data file.
        """
        self.headerparam = self.Headerparam(size, atomtypes, bondtypes, angletypes)

    def printheader(self):
        """Outputs the LAMMPS data file header in a string from the parameters set in headerparam.

        Returns
        -------
        str
            The LAMMPS data file header as generated from self.headerparam.
        """
        txt = \
f'''Generated by atom.py

{self.atoms.count()} atoms
{self.headerparam.atomtypes} atom types
{self.bonds.count()} bonds
{self.headerparam.bondtypes} bond types
{self.angles.count()} angles
{self.headerparam.angletypes} angle types
2 extra bond per atom
1 extra angle per atom
12 extra special per atom

0 {self.headerparam.size:g} xlo xhi
0 {self.headerparam.size:g} ylo yhi
0 {self.headerparam.size:g} zlo zhi

'''
        return txt

    def writefile(self, fp):
        """Writes the data contained in the Data class instance to the specified file, fp, in LAMMPS data file format.

        Parameters
        ----------
        fp : file object
            The file object that the LAMMPS data file will be written to.
        """
        fp.write(self.printheader())
        fp.write(self.atoms.section())
        fp.write(self.bonds.section())
        fp.write(self.angles.section())


