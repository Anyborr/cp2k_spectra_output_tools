#!/usr/bin/env python3

import argparse
import sys
import os
import numpy as np

# --- Helper Functions ---

def read_xyz_structure(filename):
    """
    Reads the structural .xyz file.
    Returns:
        atoms: List of dicts {'index', 'element', 'coords'}
        lines: The raw lines of the file (for accurate reproduction if needed)
        comment: The comment line
    """
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        sys.stderr.write(f"Error: File not found: {filename}\n")
        sys.exit(1)

    if not lines:
        sys.stderr.write("Error: XYZ file is empty.\n")
        sys.exit(1)

    try:
        num_atoms = int(lines[0].strip())
    except ValueError:
        sys.stderr.write("Error: Invalid number of atoms on line 1.\n")
        sys.exit(1)

    if len(lines) < num_atoms + 2:
        sys.stderr.write(f"Error: XYZ file incomplete. Expected {num_atoms + 2} lines, found {len(lines)}.\n")
        sys.exit(1)

    comment = lines[1].strip()
    atoms = []
    
    # Parse structural data
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

    # for atom in atoms:
    #     sys.stderr.write(f"atom indices: {atom['index']}, {atom['element']}\n")
    # sys.stderr.write(f"nr_atoms: {len(atoms)}\n")
    return atoms, lines, comment

def read_mask_file(filename, ref_atoms):
    """
    Reads an auxiliary file to be masked.
    Performs sanity checks against the reference atoms.
    Returns:
        raw_lines: List of strings (the lines describing the atoms)
        comment: The comment line from this file
    """
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        sys.stderr.write(f"Error: Mask file not found: {filename}\n")
        sys.exit(1)

    # Sanity Check 1: Number of atoms
    try:
        num_atoms = int(lines[0].strip())
    except ValueError:
        sys.stderr.write(f"Error: Invalid header in mask file '{filename}'.\n")
        sys.exit(1)

    if num_atoms != len(ref_atoms):
        sys.stderr.write("Error: Atom count mismatch!\n")
        sys.stderr.write(f"  Reference file: {len(ref_atoms)} atoms\n")
        sys.stderr.write(f"  Mask file '{filename}': {num_atoms} atoms\n")
        sys.exit(1)

    atom_lines = []
    
    # Sanity Check 2: Atom Types
    for i in range(num_atoms):
        line_idx = i + 2
        line = lines[line_idx]
        parts = line.strip().split()
        
        if not parts:
             sys.stderr.write(f"Error: Empty line in mask file '{filename}' at line {line_idx+1}.\n")
             sys.exit(1)

        element = parts[0]
        ref_element = ref_atoms[i]['element']

        if element != ref_element:
            sys.stderr.write(f"Error: Atom type mismatch at index {i} (Line {line_idx+1})!\n")
            sys.stderr.write(f"  Reference: {ref_element}\n")
            sys.stderr.write(f"  Mask file '{filename}': {element}\n")
            sys.exit(1)
            
        atom_lines.append(line)

    return atom_lines, lines[1].strip()

def calculate_distance(coord1, coord2):
    return np.linalg.norm(coord1 - coord2)

def identify_water_molecules(atoms, max_oh_dist=1.2):
    water_o_indices = []
    water_h_map = {}
    all_water_atoms = set()
    
    oxygens = [atom for atom in atoms if atom['element'] == 'O']
    hydrogens = [atom for atom in atoms if atom['element'] == 'H']
    h_data = [(h['index'], h['coords']) for h in hydrogens]

    sys.stderr.write(f"nr oxygens: {len(oxygens)}\n")
    sys.stderr.write(f"nr hydrogens: {len(hydrogens)}\n")
    
    for o_atom in oxygens:
        o_idx = o_atom['index']
        o_coords = o_atom['coords']
        close_h_indices = []
        for h_idx, h_coords in h_data:
            dist = calculate_distance(o_coords, h_coords)
            if dist <= max_oh_dist:
                close_h_indices.append(h_idx)
            # if o_idx == 26:    
            #     sys.stderr.write(f"o_idx 26 (dist): {dist}, h_idx: {h_idx}\n")
   
        
        if len(close_h_indices) == 2:
            water_o_indices.append(o_idx)
            water_h_map[o_idx] = close_h_indices
            all_water_atoms.add(o_idx)
            for h_i in close_h_indices:
                all_water_atoms.add(h_i)
        # else:
        #     sys.stderr.write(f"close_h_indices: {close_h_indices}\n")

    # sys.stderr.write(f"water_o_indices: {water_o_indices}\n")
    # sys.stderr.write(f"water_h_map: {water_h_map}\n")
    # sys.stderr.write(f"all_water_atoms: {all_water_atoms}\n")
    return water_o_indices, water_h_map, all_water_atoms

def parse_keep_list(input_list):
    keep_indices = set()
    if not input_list:
        return keep_indices
    for item in input_list:
        if '-' in item:
            try:
                start_s, end_s = item.split('-')
                for i in range(int(start_s), int(end_s) + 1):
                    keep_indices.add(i)
            except ValueError:
                sys.stderr.write(f"Warning: Could not parse range '{item}'.\n")
        else:
            try:
                keep_indices.add(int(item))
            except ValueError:
                sys.stderr.write(f"Warning: Could not parse index '{item}'.\n")
    return keep_indices

def main():
    parser = argparse.ArgumentParser(
        description="Extracts the closest water molecules to a center atom and applies the mask to other files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('filename', type=str, help="Path to input .xyz file")
    parser.add_argument('-c', '--center', type=int, required=True, help="Index of center atom (0-indexed)")
    parser.add_argument('-n', '--number-h2o', type=int, required=True, help="Number of water molecules to keep")
    parser.add_argument('-k', '--keep-list', nargs='+', default=[], help="List of atom indices to always keep (e.g. '1 5 10-20')")
    parser.add_argument('-m', '--mask-files', nargs='+', default=[], help="Additional files to apply the same removal mask to.")
    
    args = parser.parse_args()

    # --- Step 0: Validate Inputs ---
    # Check for duplicate filenames
    all_files = [args.filename] + args.mask_files
    if len(all_files) != len(set(all_files)):
        sys.stderr.write("Error: Duplicate filenames detected in arguments. Input and mask files must be unique.\n")
        sys.exit(1)

    # --- Step 1: Read Reference Structure ---
    sys.stderr.write(f"Reading reference structure: {args.filename}\n")
    atoms, ref_raw_lines, ref_comment = read_xyz_structure(args.filename)
    
    if not (0 <= args.center < len(atoms)):
        sys.stderr.write(f"Error: Center index {args.center} out of bounds.\n")
        sys.exit(1)

    # --- Step 2: Read and Verify Mask Files ---
    mask_files_data = []
    if args.mask_files:
        sys.stderr.write(f"Checking {len(args.mask_files)} mask files for consistency...\n")
        for m_file in args.mask_files:
            # this returns just the atom lines and the comment
            m_atom_lines, m_comment = read_mask_file(m_file, atoms)
            mask_files_data.append({
                'filename': m_file,
                'atom_lines': m_atom_lines,
                'comment': m_comment
            })
        sys.stderr.write("All mask files passed sanity checks (atom count and types match).\n")

    # --- Step 3: Solvation Shell Logic ---
    center_atom = atoms[args.center]
    user_keep_indices = parse_keep_list(args.keep_list)
    
    sys.stderr.write("--- Processing Solvation Shell ---\n")
    sys.stderr.write(f"Center Atom: {args.center} ({center_atom['element']})\n")

    water_o_indices, water_h_map, all_water_atoms = identify_water_molecules(atoms, max_oh_dist=1.2)
    
    center_coords = center_atom['coords']
    o_distances = []
    
    for o_idx in water_o_indices:
        o_coords = atoms[o_idx]['coords']
        dist = calculate_distance(center_coords, o_coords)
        o_distances.append((dist, o_idx))

    o_distances.sort(key=lambda x: x[0])
    
    target_water_count = min(args.number_h2o, len(o_distances))
    sys.stderr.write(f"target_water_count: {target_water_count}\n")  # OK
    n_closest_tuples = o_distances[:target_water_count]
    sys.stderr.write(f"n_closest_tuples: {n_closest_tuples}\n")  # OK
    sys.stderr.write(f"nbr_closest_tuples: {len(n_closest_tuples)}\n")  # OK
    closest_water_o_indices = {x[1] for x in n_closest_tuples}

    final_radius = 0.0
    if n_closest_tuples:
        max_o_dist = n_closest_tuples[-1][0]
        max_oh_bond = 0.0
        # for o_idx in closest_water_o_indices:
        #     o_coords = atoms[o_idx]['coords']
        #     for h_idx in water_h_map[o_idx]:
        #         h_coords = atoms[h_idx]['coords']
        #         oh_dist = calculate_distance(o_coords, h_coords)
        #         if oh_dist > max_oh_bond:
        #             max_oh_bond = oh_dist     
        final_radius = max_o_dist + max_oh_bond

    # --- Step 4: Determine Kept Indices ---
    kept_atom_indices = set()
    kept_atom_indices.add(args.center)
    kept_atom_indices.update(user_keep_indices)

    # Water Logic
    for o_idx in water_o_indices:
        o_atom = atoms[o_idx]
        dist_o = calculate_distance(center_coords, o_atom['coords'])
        should_keep = (o_idx in closest_water_o_indices) or (dist_o <= final_radius)
        
        if should_keep:
            kept_atom_indices.add(o_idx)
            for h_idx in water_h_map[o_idx]:
                kept_atom_indices.add(h_idx)

    # Other Atoms Logic
    for atom in atoms:
        idx = atom['index']
        if idx in kept_atom_indices: continue
        if idx in all_water_atoms: continue # Fate already decided
            
        dist = calculate_distance(center_coords, atom['coords'])
        if dist <= final_radius:
            kept_atom_indices.add(idx)

    # Convert set to sorted list for ordered output
    sorted_kept_indices = sorted(list(kept_atom_indices))
    
    # --- Step 5: Tally and Output Statistics ---
    removed_atoms_tally = {}
    for atom in atoms:
        if atom['index'] not in kept_atom_indices:
            el = atom['element']
            removed_atoms_tally[el] = removed_atoms_tally.get(el, 0) + 1

    sys.stderr.write(f"\nFinal Radius: {final_radius:.4f} A\n")
    sys.stderr.write("Deleted atoms:\n")
    for el, count in sorted(removed_atoms_tally.items()):
        sys.stderr.write(f"  {el}: {count}\n")
    sys.stderr.write("--------------------------\n")

    # --- Step 6: Output Main File to STDOUT ---
    sys.stdout.write(f"{len(sorted_kept_indices)}\n")
    new_comment = f"{ref_comment} | Solvation Shell N={target_water_count}, R={final_radius:.4f}"
    sys.stdout.write(f"{new_comment}\n")
    
    # We use the raw lines from the input file to preserve formatting
    for idx in sorted_kept_indices:
        # Original file lines start at index 2 (0=N, 1=Comment)
        sys.stdout.write(ref_raw_lines[idx + 2])

    # --- Step 7: Process and Save Mask Files ---
    if mask_files_data:
        sys.stderr.write("\nWriting masked auxiliary files:\n")
        
        for m_data in mask_files_data:
            orig_name = m_data['filename']
            # Determine new filename
            root, ext = os.path.splitext(orig_name)
            new_name = f"MASKED_{root}{ext}"
            
            try:
                with open(new_name, 'w') as f:
                    # Write Header
                    f.write(f"{len(sorted_kept_indices)}\n")
                    f.write(f"{m_data['comment']}\n")
                    
                    # Write only kept lines
                    # m_data['atom_lines'] is a list where index 0 corresponds to atom index 0
                    for atom_idx in sorted_kept_indices:
                        f.write(m_data['atom_lines'][atom_idx])
                
                sys.stderr.write(f"  -> Created: {new_name}\n")
            except IOError as e:
                sys.stderr.write(f"Error writing file {new_name}: {e}\n")

if __name__ == "__main__":
    main()
    
