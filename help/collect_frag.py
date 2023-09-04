from rdkit import Chem
from rdkit.Chem import AllChem
import networkx as nx


def show_mol(mol):
    from rdkit.Chem import Draw, AllChem
    AllChem.Compute2DCoords(mol)
    for atom in mol.GetAtoms():
        atom.SetIntProp('molAtomMapNumber', atom.GetIdx())
    Chem.MolToSmiles(mol, kekuleSmiles=True)
    a = Draw.MolToImage(mol, size=(350, 350))
    return a

def find_connect_bond(mol, clusters):
    bond = []
    for clu in clusters:
        for idx in clu:
            atom = mol.GetAtomWithIdx(idx)
            nei = list(atom.GetNeighbors())
            for n in nei:
                if n.GetIdx() not in clu:
                    bond.append([idx, n.GetIdx()])
    return bond

def linker_detect(mol, frag_id):
    atom_index = set(i for i in range(len(mol.GetAtoms())))
    frag_id_read = set(sum(frag_id, [])) # set(frag_id)
    atom_index.difference_update(frag_id_read)
    # 仅寻找构成非环键的原子以及三或四元环linker原子，即去除五元及以上的环

    tmp_idx = []
    for at_idx in atom_index:
        atom = mol.GetAtomWithIdx(at_idx)
        if atom.IsInRingSize(5) or atom.IsInRingSize(6) or atom.IsInRingSize(7) or atom.IsInRingSize(8):
            tmp_idx.append(at_idx)
    for i in tmp_idx:
        atom_index.remove(i)
    atom_index = list(atom_index)
    G = nx.Graph()
    G.add_nodes_from(atom_index)

    q = []
    for bond in mol.GetBonds():
        b = bond.GetBeginAtom().GetIdx()
        e = bond.GetEndAtom().GetIdx()
        if b in atom_index and e in atom_index:
            q.append((b, e))
    for i in q:
        G.add_edges_from([i])
    cluster = [list(i) for i in nx.connected_components(G)]  # 将相连的环归为一组

    tmp_frag_id = frag_id
    tmp_frag_id.extend(cluster)
    # print(frag_id_read)
    # print(tmp_frag_id)
    # print(Chem.MolToSmiles(mol))
    bonds = find_connect_bond(mol, tmp_frag_id)
    bonds_copy = list(set(sum(bonds, [])))
    for clu in cluster:
        same = set(bonds_copy).intersection(set(clu))
        if len(same) > 1:   # linker连接原子大于1
            frag_id.append(clu)
    return frag_id

def sanity_check_for_partial(mol, frag_id):
    frag_id = list(set(sum(frag_id, [])))
    select = Chem.MolFragmentToSmiles(mol, frag_id, isomericSmiles=False)
    a = select.count('.')
    if a != 0:
        return True
    return False

def extend_philic(mol, ids):
    tmp_hit = []
    tmp_hit.extend(ids)
    philic = []
    for idx in ids:
        a = mol.GetAtomWithIdx(idx)
        if a.IsInRing():
            continue
        philic.append(idx)

    for a in philic:
        atom = mol.GetAtomWithIdx(a)
        neighbors = list(atom.GetNeighbors())
        for n in neighbors:
            if not n.IsInRing() and n.GetIdx() not in tmp_hit and n.GetSymbol() in ['C', 'S', 'H']:
                tmp_hit.append(n.GetIdx())
                neighbors2 = list(n.GetNeighbors())
                for n2 in neighbors2:
                    if not n2.IsInRing() and n2.GetIdx() not in tmp_hit and n2.GetSymbol() in ['C', 'S', 'H']:
                        tmp_hit.append(n2.GetIdx())
    return tmp_hit

def hit_for_ring_atom(mol, idx, philic):
    ringinfo = mol.GetRingInfo()
    atomrings = list(list(items) for items in list(ringinfo.AtomRings()))
    G = nx.Graph()
    G.add_nodes_from(sum(atomrings, []))
    q = [[(s[i - 1], s[i]) for i in range(len(s))] for s in atomrings]
    for i in q:
        G.add_edges_from(i)
    rings = [list(i) for i in nx.connected_components(G)]  # 将相连的环归为一组
    hit = []
    for r in rings:
        if idx in r:
            hit.extend(r)

    tmp_hit = []
    for h in hit:
        a = mol.GetAtomWithIdx(h)
        neighbors = list(a.GetNeighbors())
        # 拓展非环侧链
        for n in neighbors:
            if not n.IsInRing() and n.GetIdx() not in tmp_hit:
                tmp_hit.append(n.GetIdx())
                neighbors2 = list(n.GetNeighbors())
                for n2 in neighbors2:
                    if not n2.IsInRing() and n2.GetIdx() not in tmp_hit and n2.GetSymbol() in ['O', 'N', 'S']:
                        tmp_hit.append(n2.GetIdx())
                        neighbors3 = list(n2.GetNeighbors())
                        for n3 in neighbors3:
                            if not n3.IsInRing() and n3.GetIdx() not in tmp_hit:
                                tmp_hit.append(n3.GetIdx())
                    else:
                        if not n2.IsInRing() and n2.GetIdx() not in tmp_hit and n2.GetSymbol() in ['C', 'F', 'Cl', 'Br']:
                            tmp_hit.append(n2.GetIdx())
    for a in philic:
        if a in tmp_hit:
            hit.extend(tmp_hit)
    return hit

def collect_frag_index(mol, anchor_atom_id):
    # bak版本： 环系统必须在两个残基均有锚点才可加入，需配合anchor_search_for_hit_ring函数扩大搜索范围
    hit = []
    anchor_atom_id_copy = []

    if type(anchor_atom_id) == list:
        for ids in anchor_atom_id:
            anchor_atom_id_copy.extend(list(ids))
    else:
        for key, ids in anchor_atom_id.items():
            anchor_atom_id_copy.extend(ids)
    anchor_atom_id_copy = list(set(anchor_atom_id_copy))

    philic = []
    for idx in anchor_atom_id_copy:
        a = mol.GetAtomWithIdx(idx)
        if a.IsInRing():
            continue
        philic.append(idx)

    for id in anchor_atom_id_copy:
        atom = mol.GetAtomWithIdx(id)
        if atom.IsInRing():  # 环上锚点，其环及附属非环结构中必须有另一锚点
            tmp_hit = hit_for_ring_atom(mol, id, philic)
            if tmp_hit not in hit:
                hit.append(tmp_hit)

    return hit


def one_attachment(smi):
    import copy
    connect = smi.count("*")
    if connect == 1:
        m = Chem.MolFromSmiles(smi)
        smi = Chem.MolToSmiles(m, isomericSmiles=False, kekuleSmiles=True)
        return [smi]
    new_mol = set()
    mol = Chem.MolFromSmiles(smi)
    atta_idx = []
    for i in range(0, mol.GetNumAtoms()):
        if mol.GetAtomWithIdx(i).GetSymbol() == '*':
            atta_idx.append(i)

    for i in atta_idx:
        m = copy.deepcopy(mol)
        mw = Chem.RWMol(m)
        for j in atta_idx:
            if i == j:
                continue
            atom = list(mol.GetAtomWithIdx(j).GetNeighbors())
            if len(atom) > 1:    # dummy连接两个原子，可能在环中
                mw.ReplaceAtom(j, Chem.rdchem.Atom(6))
            else:
                atom = atom[0]
                if atom.GetSymbol() == 'C' and not atom.IsInRing():
                    mw.ReplaceAtom(j, Chem.rdchem.Atom(54))
                else:
                    mw.ReplaceAtom(j, Chem.rdchem.Atom(6))
        """
        nei = list(mol.GetAtomWithIdx(i).GetNeighbors())
        if len(nei) == 1:
            nei = nei[0]
            if nei.GetSymbol() == 'C' and not nei.IsInRing():
                mw.ReplaceAtom(nei.GetIdx(), Chem.rdchem.Atom(0), updateLabel=True)
                mw.RemoveAtom(i)
        """
        m = Chem.Mol(mw)
        m = Chem.DeleteSubstructs(m, Chem.MolFromSmarts("[Xe]"))
        try:
            Chem.MolToSmiles(m, isomericSmiles=False, kekuleSmiles=True)
            new_mol.add(Chem.MolToSmiles(m, isomericSmiles=False))
        except:
            # print(Chem.MolToSmiles(m, isomericSmiles=False))
            continue
    return list(new_mol)

def partial_collect_for_subgraph(mol, atom_ids):
    frag_id = collect_frag_index(mol, atom_ids)
    """
    frag_id = []
    for ids in atom_ids:
        frag_id .extend(list(ids))
    # frag_id = extend_philic(mol, frag_id)
    """
    if len(frag_id) == 0:
        return None
    if sanity_check_for_partial(mol, frag_id):
        frag_id = linker_detect(mol, frag_id)
        # return None
    frag_id = list(set(sum(frag_id, [])))
    sm = Chem.MolFragmentToSmarts(mol, frag_id, isomericSmarts=False)
    target_with_dummy = AllChem.ReplaceSidechains(mol, Chem.MolFromSmarts(sm))
    if target_with_dummy is None:
        return None
    try:
        smi = Chem.MolToSmiles(target_with_dummy, isomericSmiles=True, kekuleSmiles=True)     #
        smi = Chem.MolToSmiles(target_with_dummy, isomericSmiles=False)
        smis = one_attachment(smi)
    except:
        return None
    return smis


if __name__ == '__main__':
    mol = Chem.MolFromSmiles("Oc1cccc(Nc2nccc(-n3cnc4ccccc43)n2)c1")
    atom_ids = [[5, 6, 7, 8, 11, 12, 13, 14]]
    smi = partial_collect_for_subgraph(mol, atom_ids)
