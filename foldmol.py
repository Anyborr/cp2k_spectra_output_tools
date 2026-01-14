#!/usr/bin/env python3

"""
Transforms a molecular trajectory (.xyz format) by:
1. Translating all atoms so 'center_atom_idx' is at the origin (0, 0, 0).
2. Rotating the system so the bond between 'center_atom_idx' and 
    'bond_atom_idx' is aligned with the positive x-axis.
3. (Optional) Rotating the system so the plane formed by the three atoms 
    ('center_atom_idx', 'bond_atom_idx', 'angle_atom_idx') lies in the xy-plane.

Atom indices are 0-based.

Args:
    input_filepath (str): Path to the input .xyz file.
    output_filepath (str): Path to the output .xyz file.
    center_atom_idx (int): 0-based index of the atom to center the system on.
    bond_atom_idx (int): 0-based index of the second atom for the bond vector 
    (must be different from center_atom_idx).
    angle_atom_idx (int, optional): 0-based index of the third atom to define 
    the plane. Defaults to None.
"""



import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(usage=__doc__)
    
# Possible Input Parameters
parser.add_argument('-i', metavar='input', dest='INPUT_FILE', \
                    help='name of input file', \
                    type=str, required=True)
parser.add_argument("-o", metavar="output", dest="OUTPUT_FILE", \
                    help="name of output file", \
                    type=str, required=True)
parser.add_argument("-c", metavar="center", dest="CENTER_IDX", \
                    help="index of center atom", \
                    type=int, required=True)
parser.add_argument("-b", metavar="bond", dest="BOND_IDX", \
                    help="index of bond atom", \
                    type=int, required=True)
parser.add_argument("-a", metavar="angle", dest="ANGLE_IDX", \
                    help="index of angle atom", \
                    type=int, required=False, const=None)
args = parser.parse_args()
INPUT_FILE = args.INPUT_FILE
OUTPUT_FILE = args.OUTPUT_FILE
CENTER_IDX = args.CENTER_IDX
BOND_IDX = args.BOND_IDX
ANGLE_IDX = args.ANGLE_IDX

# --- Example Usage ---

# Define the file paths and transformation atoms (0-based indexing)
# INPUT_FILE = "CYS_100_H2O-pos-1.xyz"
# OUTPUT_FILE = "transformed.xyz"

# Atom indices for a simple H2O molecule (assuming O is 0, H1 is 1, H2 is 2)
# Center on O (0), define bond along O-H1 (1), define angle plane O-H1-H2 (2)
# CENTER_IDX = 306  # Oxygen (O)
# BOND_IDX = 305    # Hydrogen 1 (H1)
# ANGLE_IDX = 313   # Hydrogen 2 (H2)


def transform_trajectory(input_filepath, output_filepath, center_atom_idx, bond_atom_idx, angle_atom_idx=None):

    # 1. --- Input Validation ---
    if center_atom_idx == bond_atom_idx:
        raise ValueError("center_atom_idx and bond_atom_idx must be different.")

    # 2. --- File Parsing and Transformation Setup ---
    with open(input_filepath, 'r') as infile, open(output_filepath, 'w') as outfile:
        # We need to peek at the first frame to get N_atoms, or assume N_atoms is constant.
        try:
            N_atoms = int(infile.readline().strip())
        except:
            raise IOError("Could not read the number of atoms from the first line of the .xyz file.")
        
        # Rewind to the beginning to start reading frames
        infile.seek(0)
        
        # Iterating through the trajectory frame by frame
        while True:
            # Read N_atoms line (first line of a frame)
            line1 = infile.readline()
            if not line1: # End of file
                break
            
            # Read comment line (second line of a frame)
            line2 = infile.readline()
            if not line2:
                # This should not happen if the file is correctly formatted
                break 

            # Read coordinates for all atoms
            symbols = []
            coords = np.zeros((N_atoms, 3))
            
            for i in range(N_atoms):
                line = infile.readline()
                if not line:
                    break
                
                parts = line.split()
                symbols.append(parts[0])
                # Coordinates start from the second part (index 1)
                coords[i] = [float(parts[j]) for j in range(1, 4)]
            
            if len(symbols) < N_atoms:
                # Incomplete frame at the end of the file
                break 

            # --- 3. Translation (Centering) ---
            # Get the coordinates of the atom to be centered
            center_coord = coords[center_atom_idx]
            # Translate all coordinates
            translated_coords = coords - center_coord

            # --- 4. Rotation Setup: Bond along X-axis ---
            
            # Vector for the bond C -> B (Center to Bond)
            # Since C is at the origin, the translated vector is just the new position of B
            bond_vector = translated_coords[bond_atom_idx]
            bond_vector_norm = np.linalg.norm(bond_vector)

            if bond_vector_norm < 1e-6:
                raise ValueError("Center and Bond atoms are coincident, cannot define a rotation.")

            # Step 4a: Define the three basis vectors for the rotation matrix
            
            # The new X-axis (e_x) is the normalized bond vector
            e_x = bond_vector / bond_vector_norm

            # The new Z-axis (e_z) needs to be constructed.
            if angle_atom_idx is not None:
                # If the 3rd atom is specified, we define the plane.
                
                # Vector C -> A (Center to Angle)
                angle_vector = translated_coords[angle_atom_idx]
                
                # The normal to the plane C-B-A is defined by the cross product of C->B and C->A
                # which will be the basis for the new Z-axis.
                e_z_unnormalized = np.cross(bond_vector, angle_vector)
                e_z_norm = np.linalg.norm(e_z_unnormalized)

                if e_z_norm < 1e-6:
                    # C-B-A are collinear (or C-A is zero), the angle_atom doesn't define a plane.
                    print(f"Warning: Atoms {center_atom_idx}, {bond_atom_idx}, {angle_atom_idx} are collinear in frame. "
                          "System will be rotated to align the bond along X, but the angle won't be guaranteed "
                          "to be in the XY plane. (Check your system or indices).")
                    
                    # Fallback: cross e_x with a general vector not parallel to e_x (e.g., [0, 0, 1])
                    if np.abs(np.dot(e_x, [0, 0, 1])) < 0.9: # If not already close to Z-axis
                        e_z_temp_unnorm = np.cross(e_x, [0, 0, 1])
                    else: # If bond is along Z, use Y-axis
                        e_z_temp_unnorm = np.cross(e_x, [0, 1, 0])
                    e_z = e_z_temp_unnorm / np.linalg.norm(e_z_temp_unnorm)
                    
                else:
                    e_z = e_z_unnormalized / e_z_norm
            
            else:
                # If no angle atom, we just ensure the bond is along X, and the orientation 
                # around the X-axis is arbitrary. We need a Z-vector orthogonal to e_x.
                
                # Cross e_x with a general vector not parallel to e_x (e.g., [0, 0, 1])
                if np.abs(np.dot(e_x, [0, 0, 1])) < 0.9: # If e_x is not close to Z-axis
                    e_z_temp_unnorm = np.cross(e_x, [0, 0, 1])
                else: # If bond is along Z, use Y-axis
                    e_z_temp_unnorm = np.cross(e_x, [0, 1, 0])
                    
                e_z = e_z_temp_unnorm / np.linalg.norm(e_z_temp_unnorm)

            # Step 4b: Calculate the final new Y-axis (e_y) 
            # e_y must be orthogonal to both e_x and e_z, determined by cross product
            e_y = np.cross(e_z, e_x)
            # Ensure e_y is normalized (should be if e_x and e_z were orthonormal)
            e_y = e_y / np.linalg.norm(e_y)
            
            # The rotation matrix R transforms the current frame basis to the target basis (x, y, z)
            # R is built by stacking the new basis vectors as *rows*
            R = np.vstack([e_x, e_y, e_z])
            
            # --- 5. Apply Rotation ---
            # Rotated coordinates are calculated by the matrix multiplication R @ translated_coords.T
            # The result is transposed back to (N_atoms, 3) format
            rotated_coords = (R @ translated_coords.T).T

            # --- 6. Write Output Frame ---
            outfile.write(f"{N_atoms}\n")
            outfile.write(line2) # Write the comment line
            
            for i in range(N_atoms):
                x, y, z = rotated_coords[i]
                # Use standard string formatting for XYZ format
                outfile.write(f"{symbols[i]:<3} {x:15.8f} {y:15.8f} {z:15.8f}\n")

    print(f"✅ Trajectory successfully transformed and saved to: {output_filepath}")

    

try:
    transform_trajectory(INPUT_FILE, OUTPUT_FILE, CENTER_IDX, BOND_IDX, ANGLE_IDX)
    
except Exception as e:
    print(f"An error occurred during transformation: {e}")
    
