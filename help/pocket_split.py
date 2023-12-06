from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.AllChem import  EmbedMolecule
import math
import networkx as nx


def lig_cor(mol):
    block = Chem.MolToPDBBlock(mol).split("\n")
    lines = []
    for line in block:
        if line.startswith("HETATM"):
            lines.append(line)
    return lines

def euclidist(A, B):
    return math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(A, B)]))

def anchor_search(mol, pocket):
    cor_mol = lig_cor(mol)
    atoms = mol.GetAtoms()
    if len(cor_mol) != len(atoms):
        raise ValueError("捕获坐标数与原子数不一致")

    anchor_atom_id = dict()
    for p in range(len(pocket)):
        dis = []
        p_cor = pocket[['x', 'y', 'z']].iloc[p].values.flatten().tolist()
        for i in range(len(atoms)):
            inf = cor_mol[i].split()
            a_cor = list(map(float, inf[5:8]))
            d = euclidist(p_cor, a_cor)
            dis.append(d)
        seq = pocket[['resSeq']].iloc[p].values.tolist()[0]
        ind = []

        for m_d in dis:
            if m_d <= 4:
                ind.append(dis.index(m_d))
        anchor_atom_id[seq] = ind
        """
        for m_d in dis:
            if seq in [186, 435, 439, 93, 436]:
                if m_d <= 4:
                    ind.append(dis.index(m_d))
            else:
                if m_d <= 4:
                    ind.append(dis.index(m_d))
        """
        anchor_atom_id[seq] = ind
    return anchor_atom_id

def hit_for_ring_atom(mol, idx):
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
    hit.extend(tmp_hit)
    return hit

def anchor_search_for_hit_ring(mol, tmp_hit):
    """先暂停使用, 弃用后收集的片段增加"""
    extended_tmp_hit = []
    extended_tmp_hit.extend(tmp_hit)
    for hit in tmp_hit:
        neighbors = list(mol.GetAtomWithIdx(hit).GetNeighbors())
        for n in neighbors:
            n_id = n.GetIdx()
            if n_id not in extended_tmp_hit:
                extended_tmp_hit.append(n_id)
            neighbors_2 = list(mol.GetAtomWithIdx(n_id).GetNeighbors())
            for n2 in neighbors_2:
                n2_id = n2.GetIdx()
                if n2_id not in extended_tmp_hit:
                    extended_tmp_hit.append(n_id)
    return extended_tmp_hit


def collect_frag_index(mol, anchor_atom_id, non_ring_id=False):
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

    non_ring = []
    for id in anchor_atom_id_copy:
        atom = mol.GetAtomWithIdx(id)
        if atom.IsInRing():  # 环上锚点，其环及附属非环结构中必须有另一锚点
            # todo partial_collect_for_subgraph 提取 atomid时，这一步骤可能会忽略 Hydrophilic 特征原子
            tmp_hit = hit_for_ring_atom(mol, id)
            if tmp_hit not in hit:
                hit.append(tmp_hit)
        else:
            non_ring.append(id)
    if non_ring_id:
        return hit.extend(non_ring)
    return hit

def delete_attachment(mol):
    mw = Chem.RWMol(mol)
    atta_idx = []
    for i in range(0, mw.GetNumAtoms()):
        if mw.GetAtomWithIdx(i).GetSymbol() == '*':
            atta_idx.append(i)
    for i in reversed(atta_idx):
        # mw.ReplaceAtom(i, Chem.rdchem.Atom(6))
        mw.ReplaceAtom(i, Chem.rdchem.Atom(1))  # 会出现化合价问题
        # mw.RemoveAtom(i)
    return Chem.Mol(mw)


def max_dummy(smi, pocket):
    """创建dummy原子"""
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol, addCoords=True)
    cor_mol = lig_cor(mol)
    atoms = mol.GetAtoms()
    if len(cor_mol) != len(atoms):
        raise ValueError("捕获坐标数与原子数不一致")

    atta_idx = []
    for i in range(0, mol.GetNumAtoms()):
        if mol.GetAtomWithIdx(i).GetSymbol() == 'H':
            atta_idx.append(i)

    dis = []
    p_cor = pocket[pocket["resSeq"] == 163][['x', 'y', 'z']].values.flatten().tolist()
    for i in atta_idx:
        inf = cor_mol[i].split()
        a_cor = list(map(float, inf[5:8]))
        d = euclidist(p_cor, a_cor)
        dis.append(d)
    dis_sort = sorted(dis)
    dis_sort.reverse()
    for d in dis_sort:
        idx = atta_idx[dis.index(d)]
        if list(mol.GetAtomWithIdx(idx).GetNeighbors())[0].IsInRing():
            mw = Chem.RWMol(mol)
            mw.ReplaceAtom(idx, Chem.rdchem.Atom(6))
            Chem.SanitizeMol(mw)
            mw = Chem.AddHs(mw, addCoords=True)
            neighbors = list(mw.GetAtomWithIdx(idx).GetNeighbors())
            for n in neighbors:
                if n.GetSymbol == "H":
                    mw.ReplaceAtom(n.GetIdx(), Chem.rdchem.Atom(0))
                    smi = Chem.MolToSmiles(Chem.Mol(mw),  isomericSmiles=True)
                    return smi
        else:
            mw = Chem.RWMol(mol)
            mw.ReplaceAtom(idx, Chem.rdchem.Atom(0))
            smi = Chem.MolToSmiles(Chem.Mol(mw), isomericSmiles=True)
            return smi

    """
    from rdkit.Chem import PyMol
    v = PyMol.MolViewer()
    v.DeleteAll()
    v.ShowMol(mol_embed, name="ori", showOnly=False)
    v.ShowMol(mol_dummy, name="lig1", showOnly=False)
    """

def sanity_check(mol, frag_id):
    frag_id = list(set(sum(frag_id, [])))
    select = Chem.MolFragmentToSmiles(mol, frag_id, isomericSmiles=False)
    a = select.count('.')
    if a != 0:
        return True
    return False

def show_mol(mol):
    from rdkit.Chem import Draw, AllChem
    AllChem.Compute2DCoords(mol)
    for atom in mol.GetAtoms():
        atom.SetIntProp('molAtomMapNumber', atom.GetIdx())
    Chem.MolToSmiles(mol, kekuleSmiles=True)
    a = Draw.MolToImage(mol, size=(350, 350))
    return a

def sanity_check2(mol):
    select = Chem.MolToSmiles(mol, isomericSmiles=False)
    a = select.count('.')
    if a != 0:
        return True
    return False

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
    # todo 还不完善，有的情况会把不相关得链返回，待研究
    atom_index = set(i for i in range(len(mol.GetAtoms())))
    frag_id_read = set(sum(frag_id, []))
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
    bonds = find_connect_bond(mol, tmp_frag_id)
    bonds_copy = list(set(sum(bonds, [])))
    for clu in cluster:
        same = set(bonds_copy).intersection(set(clu))
        if len(same) > 1:   # linker连接原子大于1
            frag_id.append(clu)
    return frag_id


def beyond_detect():
    ssr = [tuple(x) for x in Chem.GetSymmSSSR(mol)]
    anchor_atom_id = anchor_search(mol, poc)
    anchor_atom = set(sum(anchor_atom_id.values(), []))
    for s in ssr:
        if set(s).intersection(anchor_atom):
            continue
        else:
            return mol, True
    # todo 过滤碳链过长化合物，含羧基化合物

def full_collect(mol, poc):
    anchor_atom_id = anchor_search(mol, poc)
    frag_id = collect_frag_index(mol, anchor_atom_id)  # 返回包含多个环系统列表的列表
    if len(frag_id) == 0:
        return mol, False
    if sanity_check(mol, frag_id):
        frag_id = linker_detect(mol, frag_id)
    if sanity_check(mol, frag_id):
        return mol, False
    frag_id = list(set(sum(frag_id, [])))
    sm = Chem.MolFragmentToSmarts(mol, frag_id, isomericSmarts=False)
    target_with_dummy = AllChem.ReplaceSidechains(mol, Chem.MolFromSmarts(sm))
    mol_dummy = delete_attachment(target_with_dummy)
    try:
        Chem.SanitizeMol(mol_dummy)
    except:
        return mol, False
    if sanity_check2(mol_dummy):
        return mol, False
    mol_dummy = Chem.RemoveHs(mol_dummy)
    mol_dummy = Chem.AddHs(mol_dummy, addCoords=True)
    return mol_dummy, True

def embed(mol, mol_dummy, sm):
    # 无法在截取的片段分子上另外添加片段生成3D构象，该函数无法使用
    chain = Chem.MolFromSmiles('CC(=O)N(C1)Cc(c12)nc(C)cc2C')
    replace = Chem.rdmolops.ReplaceSubstructs(mol, Chem.MolFromSmiles('*'), chain, replacementConnectionPoint=0)
    Chem.SanitizeMol(replace[0])
    core = Chem.MolFromSmarts(sm)
    match = replace[0].GetSubstructMatch(core)
    if not match:
        return None
    # coordMap = {}
    # coreConf = replace[0].GetConformer()
    # for i1, idxI in enumerate(match):
    #     corePtI = coreConf.GetAtomPosition(i1)
    #     coordMap[idxI] = corePtI
    mol_2 = Chem.AddHs(replace[0])
    cid = EmbedMolecule(mol_2, maxAttempts=1000, clearConfs=True, ignoreSmoothingFailures=False)
    while cid != 0:
        cid = EmbedMolecule(mol_2, maxAttempts=1000, clearConfs=True, ignoreSmoothingFailures=False)
    # 不能使用coordMap=coordMap，否则会产生报错 Could not triangle bounds smooth molecule.
    # cids = EmbedMultipleConfs(mol_2, numConfs=1, coordMap=coordMap, numThreads=0)
    # block = Chem.MolToPDBBlock(mol_2).split("\n")
    prbMatch = mol.GetSubstructMatch(core)
    refMatch = mol_dummy.GetSubstructMatch(core)
    AllChem.AlignMol(mol_2, mol_dummy, atomMap=list(zip(prbMatch, refMatch)))
    return mol_2

# fs = cut(mol); Draw.MolsToImage(fs).show()
def cut(mol):
    """from chemistGA"""
    import random
    if not mol.HasSubstructMatch(Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')):
        return None
    bis = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')))  # single bond not in ring
    # print bis,bis[0],bis[1]
    bs = [mol.GetBondBetweenAtoms(bis[0], bis[1]).GetIdx()]
    fragments_mol = Chem.FragmentOnBonds(mol, bs, addDummies=True, dummyLabels=[(1, 1)])
    try:
        fragments = Chem.GetMolFrags(fragments_mol, asMols=True)
        return fragments
    except:
        return None


def partial_collect_bak(mol, poc):
    anchor_atom_id = anchor_search(mol, poc)
    frag_id = collect_frag_index(mol, anchor_atom_id, non_ring_id=True)    # 小修改
    if len(frag_id) == 0:
        return mol, False
    if sanity_check(mol, frag_id):
        frag_id = linker_detect(mol, frag_id)
    frag_id = list(set(sum(frag_id, [])))
    sm = Chem.MolFragmentToSmarts(mol, frag_id, isomericSmarts=False)
    target_with_dummy = AllChem.ReplaceSidechains(mol, Chem.MolFromSmarts(sm))
    mol_dummy = delete_attachment(target_with_dummy)
    try:
        Chem.SanitizeMol(mol_dummy)
    except:
        return mol, False
    if sanity_check2(mol_dummy):
        return mol, False
    # 同full_collect开始不同

    mol_du = max_dummy(mol_dummy, poc)
    if mol_du is None:
        return mol, False

    mol_du = Chem.RemoveHs(mol_du)
    mol_embed = embed(mol_du, mol_dummy, sm)
    if mol_embed is None:
        return mol, False
    return mol_embed, True


def partial_collect(mol, poc):
    anchor_atom_id = anchor_search(mol, poc)
    frag_id = collect_frag_index(mol, anchor_atom_id)
    if len(frag_id) == 0:
        return None, None, None
    if sanity_check(mol, frag_id):
        frag_id = linker_detect(mol, frag_id)
    frag_id = list(set(sum(frag_id, [])))
    sm = Chem.MolFragmentToSmarts(mol, frag_id, isomericSmarts=False)
    target_with_dummy = AllChem.ReplaceSidechains(mol, Chem.MolFromSmarts(sm))
    smi = Chem.MolToSmiles(target_with_dummy, isomericSmiles=False)     # , isomericSmiles=True, kekuleSmiles=True    kekule无问题
    # isomericSmiles=False 此时输出的分子无括号，dummy原子无序号
    mol_dummy = delete_attachment(target_with_dummy)
    try:
        Chem.SanitizeMol(mol_dummy)
    except:
        return None, None, None
    if sanity_check2(mol_dummy):
        return None, None, None
    mol_dummy = Chem.RemoveHs(mol_dummy)
    mol_dummy = Chem.AddHs(mol_dummy, addCoords=True)
    return mol_dummy, smi, frag_id

def remain(mol, frag_id):
    atom_index = set(i for i in range(len(mol.GetAtoms())))
    remain_id = atom_index.difference(set(frag_id))
    frag_id = collect_frag_index(mol, [remain_id])
    if len(frag_id) == 0:
        return None, None
    if sanity_check(mol, frag_id):
        frag_id = linker_detect(mol, frag_id)
    frag_id = list(set(sum(frag_id, [])))

    sm = Chem.MolFragmentToSmarts(mol, frag_id, isomericSmarts=False)
    target_with_dummy = AllChem.ReplaceSidechains(mol, Chem.MolFromSmarts(sm))
    smi = Chem.MolToSmiles(target_with_dummy, isomericSmiles=False)    # , kekuleSmiles=True
    mol_dummy = delete_attachment(target_with_dummy)     # remain中dummy原子可能不为末端原子，dummy原子连接两个原子
    # smi = Chem.MolToSmiles(mol_dummy, isomericSmiles=True)
    try:
        Chem.SanitizeMol(mol_dummy)
    except:
        return None, None
    if sanity_check2(mol_dummy):
        return None, None
    mol_dummy = Chem.RemoveHs(mol_dummy)
    mol_dummy = Chem.AddHs(mol_dummy, addCoords=True)
    return mol_dummy, smi


def collect(mol, frag_id):
    if sanity_check(mol, frag_id):
        frag_id = linker_detect(mol, frag_id)
    frag_id = list(set(sum(frag_id, [])))
    sm = Chem.MolFragmentToSmarts(mol, frag_id, isomericSmarts=False)
    target_with_dummy = AllChem.ReplaceSidechains(mol, Chem.MolFromSmarts(sm))
    if target_with_dummy is None:
        return None
    smis = Chem.MolToSmiles(target_with_dummy, isomericSmiles=True, kekuleSmiles=True).split(".")
    #  isomericSmiles=True SMILES会保留dummylabel
    # from rdkit.Chem import Draw
    # AllChem.Compute2DCoords(target_with_dummy); Draw.MolToImage(target_with_dummy).show()
    smi = [x for _, x in sorted(zip(map(len, smis), smis), reverse=True)][0]
    return smi



if __name__ == '__main__':

    # one molecule test
    mol = Chem.SDMolSupplier("data/irg1_chembl/chembl_549986.sdf", removeHs=True)[0]
    poc = load_rec("data/irg1_chembl/IRG1.pdb")
    anchor_atom_id = anchor_search(mol, poc)
    show_mol(mol).show()  # 会抹除对接构象
    frag, smi, frag_id = partial_collect(mol, poc)


    """
    # $sch/utilities/glide_ensemble_merge -osd -u s_m_title vsw_chembl1-SP_OUT_1_pv.maegz vsw_chembl2_2-SP_OUT_1_pv.maegz vsw_chembl3-SP_OUT_1_pv.maegz vsw_chembl4-SP_OUT_1_pv.maegz
    # mol = Chem.MolFromSmiles("Cc(n1)cc(C)c(c12)CN(C2)C(=O)CC3CN(C3)c4ccnc(C(F)(F)F)c4")
    suppl1 = Chem.SDMolSupplier("../data/m4_chefliter/vsw_chemfliter.sdf", removeHs=True)   # 58172
    suppl1.SetProcessPropertyLists(False)
    suppl = [suppl1]
    writer = Chem.SDWriter('../data/m4_chefliter/vsw_chemfliter_ps.sdf')
    error_writer = Chem.SDWriter('../data/m4_chefliter/vsw_chemfliter_ps_error.sdf')

    poc = load_rec_bak()
    count = 0
    for i, su in enumerate(suppl, 1):
        for mol in tqdm(su, desc=f"{i}th file process "):
            if mol is None:
                continue
            frag, state = full_collect(mol, poc)
            # frag, state = partial_collect(mol, poc)       # 应用时需修改口袋定位残基
            if not state:
                error_writer.write(frag)
            else:
                ori_smi = Chem.MolToSmiles(mol, kekuleSmiles=True)
                frag.SetProp('ori_smi', f'{ori_smi}')
                writer.write(frag)
                count += 1
    print(f"{count} 个片段已收集")
    writer.close()
    error_writer.close()
    """
