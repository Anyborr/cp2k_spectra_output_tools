#!/usr/bin/env python3

import argparse
import sys
import numpy as np

def wrap_pbc(position, cell_length):
    """
    Applies Periodic Boundary Conditions (PBC) to a single coordinate.
    The final position is guaranteed to be in the range [0, L)
    for each dimension, where L is cell_length.

    Args:
        position (np.ndarray): The 3D coordinate [x, y, z].
        cell_length (float): The length of the cubic cell (L).

    Returns:
        np.ndarray: The wrapped 3D coordinate.
    """
    # Standard formula for Minimum Image Convention centered at origin
    wrapped_position = position - cell_length * np.round(position / cell_length)
    return wrapped_position


# def find_molecular_water(positions, atom_kinds, min_dist=1.1):
#     """
#     Finds molecular water molecules and return their indices. For use with keeping
#     molecules together after wrapping.
#     """

#     # First we identify indices of all oxygen atoms
    

def calculate_distance(coord1, coord2):
    return np.linalg.norm(coord1 - coord2)

def identify_water_molecules(atoms, max_oh_dist=1.2):
    water_o_indices = []
    water_h_map = {}
    all_water_atoms = set()
    
    oxygens = [atom for atom in atoms if atom['element'] == 'O']
    hydrogens = [atom for atom in atoms if atom['element'] == 'H']
    h_data = [(h['index'], h['coords']) for h in hydrogens]

    # sys.stderr.write(f"nr oxygens: {len(oxygens)}\n")
    # sys.stderr.write(f"nr hydrogens: {len(hydrogens)}\n")
    
    for o_atom in oxygens:
        o_idx = o_atom['index']
        o_coords = o_atom['coords']
        close_h_indices = []
        for h_idx, h_coords in h_data:
            dist = calculate_distance(o_coords, h_coords)
            if dist <= max_oh_dist:
                close_h_indices.append(h_idx)   
        
        if len(close_h_indices) == 2:
            water_o_indices.append(o_idx)
            water_h_map[o_idx] = close_h_indices
            all_water_atoms.add(o_idx)
            for h_i in close_h_indices:
                all_water_atoms.add(h_i)

    # sys.stderr.write(f"water_o_indices: {water_o_indices}\n")
    # sys.stderr.write(f"water_h_map: {water_h_map}\n")
    # sys.stderr.write(f"all_water_atoms: {all_water_atoms}\n")
    return water_o_indices, water_h_map, all_water_atoms



def process_xyz_trajectory(input_filepath, output_filepath, center_atom_index, cell_length):
    """
    Reads an XYZ file, centers the trajectory on a specified atom,
    applies PBC wrapping, and writes the output.

    Args:
        input_filepath (str): Path to the input .xyz file.
        output_filepath (str): Path to the output .xyz file.
        center_atom_index (int): Zero-indexed index of the atom to center on.
        cell_length (float): Length of the cubic PBC cell.
    """
    try:
        # This reads whole file into memory. Easy to code but Bad idea for large files !!
        with open(input_filepath, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_filepath}", file=sys.stderr)
        sys.exit(1)

    atoms = []
    mol_id = 0
    num_atoms = int(lines[0].strip())
    sys.stderr.write(f"nr atoms: {num_atoms}\n")

    # Parse structural data to identify h2o molecules in the 1st snapshot
    for i in range(2, num_atoms + 2):
        parts = lines[i].strip().split()
        if len(parts) < 4:
            sys.stderr.write(f"Error: Invalid atom line format at line {i+1}: '{lines[i].strip()}'\n")
            sys.exit(1)
        
        element = parts[0]
        try:
            coords = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
        except ValueError:
            sys.stderr.write(f"Error: Invalid coordinate format at line {i+1}.\n")
            sys.exit(1)

        atoms.append({'index': i - 2, 'element': element, 'coords': coords})

    water_o_indices, water_h_map, all_water_atoms = identify_water_molecules(atoms, max_oh_dist=1.2)
    # sys.stderr.write(f"nr_h2o oxy: {len(water_o_indices)}\n")
    # sys.stderr.write(f"water_h_map: {water_h_map}\n")


    # from here we process atoms line-by-line
    output_lines = []
    current_line = 0
    num_atoms = 0
    frame_count = 0

    print(f"Processing trajectory file: {input_filepath}")
    print(f"Output file: {output_filepath}")
    print(f"Centering atom index: {center_atom_index}")
    print(f"PBC cell length: {cell_length}")

    while current_line < len(lines):
        try:
            # 1. Read the number of atoms (N)
            num_atoms_line = lines[current_line].strip()
            if not num_atoms_line:
                current_line += 1
                continue
                
            num_atoms = int(num_atoms_line)
            output_lines.append(lines[current_line])
            current_line += 1
            frame_count += 1
            
            # Check for valid center atom index
            if center_atom_index >= num_atoms or center_atom_index < 0:
                print(f"Error in frame {frame_count}: Atom index {center_atom_index} is out of range (0 to {num_atoms - 1}).", file=sys.stderr)
                sys.exit(1)

            # 2. Read the comment line
            comment_line = lines[current_line]
            output_lines.append(comment_line)
            current_line += 1

            # 3. Read atom coordinates for the current frame
            frame_data = []
            for i in range(num_atoms):
                parts = lines[current_line].strip().split()
                if not parts or len(parts) < 4:
                    raise ValueError("Incomplete or improperly formatted atom data line.")
                
                atom_symbol = parts[0]
                # Coordinates start from the second element (index 1)
                coords = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                atom_index = (current_line % (num_atoms+2)) -2
                frame_data.append({'symbol': atom_symbol, 'coords': coords, 'index': atom_index})
                current_line += 1
            #sys.stderr.write(f"frame_data: {frame_data}\n")
                
        except IndexError:
            # This handles unexpected end of file
            print(f"Warning: Trajectory file ended unexpectedly after reading {frame_count} frames.", file=sys.stderr)
            break
        except ValueError as e:
            print(f"Error reading coordinates in frame {frame_count} at line {current_line}. Details: {e}", file=sys.stderr)
            sys.exit(1)
            
        # --- PROCESSING STEP ---

        # Determine the position of the atom to center on
        center_position = frame_data[center_atom_index]['coords']

        # Define translation point w.r.t <0, 0, 0>
        translation_vector = np.array([cell_length/2, cell_length/2, cell_length/2])

        translated_atoms = []
        # Apply the translation and PBC wrapping
        for atom in frame_data:
            # Step 1: Translate all atoms so the center atom's new position is <0, 0, 0>
            translated_coords = atom['coords'] - center_position

            # Step 2: Apply Periodic Boundary Conditions (PBC) wrapping
            wrapped_coords = wrap_pbc(translated_coords, cell_length)

            # Step 3: Translate by L/2 so center atom is at center of box
            centered_coords = wrapped_coords + translation_vector

            translated_atoms.append({'symbol': atom['symbol'], 'index': atom['index'], 'coords': centered_coords})
            
        # move atoms together
        # We do this in a separate loop to avoid having hydrogens moved again afterwards
        for atom in translated_atoms:
            
            # Step 4: Move hydrogens of each water molecule to their respective oxygens
            atom_idx = atom['index']
            if atom_idx in water_o_indices:
                for h_atom in water_h_map[atom_idx]:
                    h_coords = translated_atoms[h_atom]['coords']
                    o_coords = translated_atoms[atom_idx]['coords']
                    dist = calculate_distance(o_coords, h_coords)
                    if dist > cell_length/2:
                        h_translate_vector = np.round((o_coords-h_coords) / cell_length) * cell_length
                        # sys.stderr.write(f"h_translate_vector: {h_translate_vector}\n")
                        # sys.stderr.write(f"dist: {dist}\n")
                        h_coords_translated = h_coords + h_translate_vector
                        translated_atoms[h_atom]['coords'] = h_coords_translated
            
        for atom in translated_atoms:
            # Prepare the output line (formatted to 6 decimal places)
            output_line = f"{atom['symbol']} {atom['coords'][0]:.6f} {atom['coords'][1]:.6f} {atom['coords'][2]:.6f}\n"
            output_lines.append(output_line)

    # Write the output file
    try:
        with open(output_filepath, 'w') as f:
            f.writelines(output_lines)
        print(f"\nSuccessfully processed {frame_count} frames and wrote output to: {output_filepath}")
    except IOError:
        print(f"Error writing to output file {output_filepath}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Center an XYZ trajectory on a specified atom and apply PBC wrapping."
    )
    
    # -i flag for input file (required)
    parser.add_argument(
        '-i', '--input-file',
        type=str,
        required=True,
        dest='input_file',
        help="Path to the input .xyz trajectory file."
    )

    # -o flag for output file (required)
    parser.add_argument(
        '-o', '--output-file',
        type=str,
        required=True,
        dest='output_file',
        help="Path for the output .xyz trajectory file."
    )

    # -c flag (required)
    parser.add_argument(
        '-c', '--center-atom',
        type=int,
        required=True,
        dest='center_atom_index',
        help="Zero-indexed atom index to use as the center (<L/2,L/2,L/2>)."
    )
    
    # -l flag (required)
    parser.add_argument(
        '-l', '--cell-length',
        type=float,
        required=True,
        dest='cell_length',
        help="Length of the cubic Periodic Boundary Condition (PBC) cell."
    )

    args = parser.parse_args()

    process_xyz_trajectory(
        args.input_file,
        args.output_file,
        args.center_atom_index,
        args.cell_length
    )

if __name__ == "__main__":
    main()
    
