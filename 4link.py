from rdkit import Chem
from tqdm import tqdm
import networkx as nx
from help.dis_pharma import create_frag_graph
import pandas as pd
from rdkit.Chem import PandasTools
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

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
            if len(atom) > 1:
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
            new_mol.add(Chem.MolToSmiles(m, isomericSmiles=False, kekuleSmiles=True))
        except:
            print(Chem.MolToSmiles(m, isomericSmiles=False))
            continue
    return list(new_mol)

def delete_attachment_m2(smi):
    new_mol = set()
    mol = Chem.MolFromSmiles(smi)
    atta_idx = []
    for i in range(0, mol.GetNumAtoms()):
        if mol.GetAtomWithIdx(i).GetAtomicNum() == 0:
            atta_idx.append(i)

    mw = Chem.RWMol(mol)
    for i in atta_idx:
        atom = list(mol.GetAtomWithIdx(i).GetNeighbors())[0]
        if atom.GetSymbol() == 'C' and not atom.IsInRing():
            mw.ReplaceAtom(i, Chem.rdchem.Atom(54))
        else:
            mw.ReplaceAtom(i, Chem.rdchem.Atom(6))
    m = Chem.Mol(mw)
    m2 = Chem.DeleteSubstructs(m, Chem.MolFromSmarts("[Xe]"))
    new_mol.add(Chem.MolToSmiles(m2, isomericSmiles=True, kekuleSmiles=True))
    return list(new_mol)


def load_remain(remain="data/irg1_chembl/search/remain_frag.smi"):
    with open(remain, 'rt') as f:
        mols = f.readlines()
    mol = [m.strip("*\n") for m in mols]
    return mol

def create_newmols(fragments):
    import traceback
    child_set = set()
    # chains = load_remain()
    chains = ['COc1nc[nH]c1']  # CC(=O)Nc1nc[nH]n1  CC(=O)Nc1nc[nH]c1
    for smile in tqdm(fragments, desc="crossover  "):
        connect = smile.count("*")
        if connect == 0:
            child_set.add(smile)
        else:
            smis = one_attachment(smile)
            for smi in smis:
                for chain in chains:
                    chain = Chem.MolFromSmiles(chain)
                    try:
                        newfrag = Chem.MolFromSmiles(smi)
                        frag1 = Chem.rdmolops.ReplaceSubstructs(newfrag, Chem.MolFromSmiles('*'), chain,
                                                                replacementConnectionPoint=0)
                        Chem.SanitizeMol(frag1[0])
                        child_smi = Chem.MolToSmiles(frag1[0])
                        child_set.add(child_smi)
                    except:
                        continue
                        # print('--------traceback.format_exc--------')
                        # error = traceback.format_exc()
                        # print(error)
                        # print(smis, smile)
                        # raise Exception(error)
    return child_set

def Mol_Conn(s1, s2, attachment):
    """可以尝试使用， 代替create_newmols"""
    conn = []
    m1 = Chem.MolFromSmiles(s1)
    m2 = Chem.MolFromSmiles(s2)
    m = Chem.CombineMols(m1, m2)
    mw = Chem.RWMol(m)
    for i in range(0, m1.GetNumAtoms()):
        if m1.GetAtomWithIdx(i).GetSymbol() == attachment:
            neighbor1_idx = m1.GetAtomWithIdx(i).GetNeighbors()[0].GetIdx()
            for j in range(0, m2.GetNumAtoms()):
                if m2.GetAtomWithIdx(j).GetSymbol() == attachment:
                    neighbor2_idx = m2.GetAtomWithIdx(j).GetNeighbors()[0].GetIdx()
                    mw.AddBond(neighbor1_idx, neighbor2_idx + m1.GetNumAtoms(), Chem.BondType.SINGLE)
                    mw.RemoveAtom(j + m1.GetNumAtoms())
                    mw.RemoveAtom(i)
                    conn.append(Chem.MolToSmiles(mw))
                    mw = Chem.RWMol(m)
    return conn


def link(num_cpu, mols_text, name="seed_R5R6"):
    from multiprocessing import Pool, cpu_count
    ncpu = min(cpu_count(), max(num_cpu, 1))
    p = Pool(ncpu)

    # with open('data/new_graph/xp_VAL163_below8_search_zinc.csv', 'rt') as f:
    #     mols_text = f.readlines()

    batch_size = len(mols_text) // ncpu + 1
    batches = [mols_text[i: i + batch_size] for i in range(0, len(mols_text), batch_size)]

    result = set()
    for i, res in enumerate(p.imap_unordered(create_newmols, batches), 1):
        result.update(res)
        print(f'\r {i} batch collected,  {len(res)} mols linked, total {len(result)} mols completed ')
        # sys.stderr.write(f'\r {i} batch collected,  collect {len(res)} frags ')
        # sys.stderr.flush()
    print()
    print(f"{len(result)} mols completed")

    # df = pd.DataFrame.from_dict(result, orient='index', columns=['hash', 'ori_smi'])
    # df = df.reset_index().rename(columns={'index': 'smi'})
    # df.to_csv("data/irg1_chembl/search/link_search_chembl.csv", header=False, index=False)
    # with open("data/irg1_zinc/ps/xp_VAL163_below9_search_zinc_link.smi", 'wt') as f:
    #     f.write('\n'.join([f"{smile},{hash}" for smile, hash in result.items()]))  # ring_sum.keys()
    with open(f"data/irg1_chembl/search/link_{name}.smi", 'wt') as f:
        f.write('\n'.join([smile for smile in result]))  # ring_sum.keys()
    return 0


def delete_attachment_prepare(fragments):
    child_set = set()
    for smile in tqdm(fragments, desc="del    "):
        # smile, hash, ori_smi = frag.split(",")
        # ori_smi = ori_smi.strip("\n ")
        connect = smile.count("*")
        if connect == 0:
            child_set.add(smile)
        else:
            smis = one_attachment(smile)
            # smis = delete_attachment_m2(smile)
            for smi in smis:
                s = delete_attachment_m2(smi)
                for i in s:
                    child_set.add(i)
    return child_set

def delete_attachment(num_cpu, mols_text):
    from multiprocessing import Pool, cpu_count
    ncpu = min(cpu_count(), max(num_cpu, 1))
    p = Pool(ncpu)

    batch_size = len(mols_text) // ncpu + 1
    batches = [mols_text[i: i + batch_size] for i in range(0, len(mols_text), batch_size)]

    result = set()
    for i, res in enumerate(p.imap_unordered(delete_attachment_prepare, batches), 1):
        result.update(res)
        print(f'\r {i} batch collected,  {len(res)} mols linked, total {len(result)} mols completed ')
        # sys.stderr.write(f'\r {i} batch collected,  collect {len(res)} frags ')
        # sys.stderr.flush()
    print()
    print(f"{len(result)} mols completed")

    # df = pd.DataFrame.from_dict(result, orient='index', columns=['hash', 'ori_smi'])
    # df = df.reset_index().rename(columns={'index': 'smi'})
    # df.to_csv("data/new_graph/xp_VAL163_below8_search_zinc_delete_attach.csv", header=False, index=False)
    # with open("data/irg1_zinc/ps/xp_VAL163_below9_search_zinc_link.smi", 'wt') as f:
    #     f.write('\n'.join([f"{smile},{hash}" for smile, hash in result.items()]))  # ring_sum.keys()
    with open("data/irg1_chembl/search/delat_R5R6.smi", 'wt') as f:
        f.write('\n'.join([smile for smile in result]))  # ring_sum.keys()
    return 0


def ana(df, edge="distance", node="define"):
    tqdm.pandas(desc='pandas bar |   ')
    df["frag_graph&diameter"] = df["mol_object"].progress_apply(create_frag_graph)
    df["hash_distance"] = df["frag_graph&diameter"].progress_apply(
        lambda obj: nx.weisfeiler_lehman_graph_hash(obj[0], edge_attr=edge, node_attr=node))
    df["diameter"] = df["frag_graph&diameter"].progress_apply(lambda obj: obj[1])
    gscore_groups = df.groupby("hash_distance")
    gscore = dict()
    for group in gscore_groups.r_i_glide_gscore:
        group = list(group)
        scores = group[1].values.tolist()
        gscore[group[0]] = scores


def creat_dict(df, key="hash_distance", val="cpmp2"):
    res = dict()
    cur_df = df.drop_duplicates(subset=[key], keep="first")
    for i, row in cur_df.iterrows():
        name = row[key]
        target = row[val]
        res[name] = target
    return res

class MolData:
    def __init__(self, path):
        self.df = PandasTools.LoadSDF(path, molColName='mol_object', removeHs=True)


def simplify_component(component):
    com2 = ""
    com = component.split(";")
    count = 0
    for c in com:
        if c == 'Hydrophilic':
            count += 1
            continue
        if c:
            nc = c.split("_")
            com2 += f"{nc[0]};"    # _{nc[1]}
    """
    if count < 1:
        com2 += f"philic_0"
    else:
        com2 += f"philic_ge1"
    """
    return com2

if __name__ == '__main__':

    tqdm.pandas(desc='pandas bar |   ')
    new_df = pd.read_csv("data/irg1_chembl/xp_VAL163.csv")

    """
    search = pd.read_csv("data/irg1_chembl/search/VAL163_search_chembl.csv", header=None,
                         names=["smi", "hash", "ori_smi"])

    search_group = search.groupby("hash")
    sitedf = new_df.reset_index(drop=True)
    site_group = sitedf.groupby("hash_distance")

    new_search = pd.DataFrame()
    recover = pd.DataFrame()
    for group in search_group:
        g = list(group)
        name = g[0]
        df = g[1].drop_duplicates(subset=["smi"], keep="first", inplace=False).copy()
        seed_df = sitedf.iloc[site_group.groups[name].tolist()].copy()
        seed_smi = seed_df["smi"].values.tolist()
        df["save"] = df["smi"].apply(lambda smi: 1 if smi not in seed_smi else 0)
        cur_df = df[df["save"] == 1].copy()
        cur_recover = df[df["save"] == 0].copy()
        new_search = pd.concat([new_search, cur_df])
        recover = pd.concat([recover, cur_recover])

    # df = pd.read_csv("data/irg1_zinc/ps/xp_VAL163_below9_search_zinc_link.smi", header=None, names=["smi", "hash"])
    # df["mol_object"] = df["smi"].apply(Chem.MolFromSmiles)
    # PandasTools.WriteSDF(df, "data/irg1_zinc/ps/xp_VAL163_below9_search_zinc_link.sdf", molColName="mol_object", properties=["hash", "smi"])


    search_hash = new_search.value_counts(subset=["hash"])
    search_hash = search_hash.reset_index().rename(columns={'index': 'hash'})
    search_hash.rename(columns={0: "occurence"}, inplace=True)

    comp2_dict = creat_dict(new_df, key="hash_distance", val="comp2")
    search_hash["comp2"] = search_hash["hash"].progress_apply(lambda x: comp2_dict[x])
    """

    new_search = pd.read_csv("data/irg1_chembl/search/new_search.csv")
    comp2_dict = creat_dict(new_df, key="hash_distance", val="comp2")
    new_search["comp2"] = new_search["hash"].progress_apply(lambda x: comp2_dict[x])

    mols1 = new_df[new_df["comp2"] == "R5;R6;"].copy()
    mols1.drop_duplicates(subset=["smi"], keep="first", inplace=True)
    mols1_text = mols1["smi"].values.tolist()
    b = link(60, mols1_text, "seed_R5R6")
    # b = delete_attachment(60, mols_text)

    mols2 = new_search[new_search["comp2"] == "R5;R6;"].copy()
    mols2.drop_duplicates(subset=["smi"], keep="first", inplace=True)
    mols2_text = mols2["smi"].values.tolist()
    c = link(60, mols2_text, "R5R6")

    mols3 = new_df[new_df["comp2"] == "R5;R5;"].copy()
    mols3.drop_duplicates(subset=["smi"], keep="first", inplace=True)
    mols3_text = mols3["smi"].values.tolist()
    b = link(60, mols3_text, "seed_R5R5")
    # b = delete_attachment(60, mols_text)

    mols4 = new_search[new_search["comp2"] == "R5;R5;"].copy()
    mols4.drop_duplicates(subset=["smi"], keep="first", inplace=True)
    mols4_text = mols4["smi"].values.tolist()
    c = link(60, mols4_text, "R5R5")


    """
    df = PandasTools.LoadSDF("data/irg1_chembl/xp_VAL163_remain.sdf", molColName='mol_object', removeHs=True)
    smi_set = set()
    # for smile in df["smi"].values.tolist():
    for smile in r5r5["remain_smi"].values.tolist():
        if type(smile) != str:
            continue
        smis = one_attachment(smile)
        smi_set.update(set(smis))
        
    with open("data/irg1_chembl/search/remain_frag.smi", 'wt') as f:         # r'E:\database\fda_ring.smi'
        f.write('\n'.join([line for line in smi_set]))
    """
