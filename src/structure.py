from Bio.PDB import PDBList, PDBParser
import numpy as np

# 4K62: H5
# 1HGF: H3
# 3ZTN: H1
def get_seq_coordinate(subtype, chain_id, atom_id):
    if subtype == 'H1':
        pdb_id = '3ZTN'
        chain_idx = range(0, 326)
    elif subtype == 'H3':
        pdb_id = '1HGF'
        chain_idx = range(0, 328)
    elif subtype == 'H5':
        pdb_id = '4K62'
        chain_idx = range(1, 318)
    pdb_list = PDBList()
    pdb_file_name = pdb_list.retrieve_pdb_file(pdb_code=pdb_id, pdir='./metadata', file_format='pdb', overwrite=False)
    pdb_parser = PDBParser(QUIET=True)
    HA1_structure = pdb_parser.get_structure(pdb_id, pdb_file_name)
    seq_coordinate = []
    for model in HA1_structure:
        print(len(model[chain_id]))
        for idx, residue in enumerate(model[chain_id]):
            if idx not in chain_idx:
                continue
            if atom_id not in residue:
                raise ValueError(f"Atom ID {atom_id} not found in residue {residue}")
            seq_coordinate.append(residue[atom_id].get_coord())
    return np.array(seq_coordinate)