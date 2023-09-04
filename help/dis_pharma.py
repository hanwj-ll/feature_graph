import math
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures, AllChem
from .collect_frag import partial_collect_for_subgraph

factory = ChemicalFeatures.BuildFeatureFactory("help/fdef/define.fdef")


def euclidist(A, B):
    return round(math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(A, B)])), 3)

def get_point(pos):
    return pos.x, pos.y, pos.z

def find_connect_2(idx, cls):
    for c in cls:
        if idx in c:
            return c
    return None

def find_connect(atom_ids, mol):
    atom_ids_sort = sorted(atom_ids)
    connect = set()
    for n, ids1 in enumerate(atom_ids_sort):
        if len(ids1) == 1:
            continue
        for ele in ids1:
            neigh = mol.GetAtomWithIdx(ele).GetNeighbors()
            for nei in neigh:
                idx = nei.GetIdx()
                if idx in ids1:
                    continue
                c = find_connect_2(idx, atom_ids_sort)   # [n+1:]
                if c is not None:
                    connect.add(tuple(sorted([atom_ids.index(ids1), atom_ids.index(c)])))
                else:
                    neigh2 = nei.GetNeighbors()
                    for nei2 in neigh2:
                        idx2 = nei2.GetIdx()
                        if idx2 in ids1 or idx2 == idx:
                            continue
                        c = find_connect_2(idx2, atom_ids_sort)   # [n + 1:]
                        if c is not None:
                            connect.add(tuple(sorted([atom_ids.index(ids1), atom_ids.index(c)])))
    return connect

def get_node(graph):
    pharma = []
    for v, attr in graph.nodes(data=True):
        pharma.append(attr["define"])
    res = ""
    for fea in sorted(pharma):
        res += f"{fea};"
    return res

def find_bin(dis, dis_bin):
    for n, value in enumerate(dis_bin):
        if dis < value:
            return f"{n}"
    return f"{len(dis_bin) + 1}"

def create_frag_graph(mol, distance_bin=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14)):
    AllChem.Compute2DCoords(mol)
    feats = factory.GetFeaturesForMol(mol)
    feats_pharma = [(feat.GetFamily(), get_point(feat.GetPos()), feat.GetAtomIds()) for feat in feats if
                    feat.GetFamily() not in ["Aromatic_1", "Aromatic_2"]]
    feats_aromatic = [get_point(feat.GetPos()) for feat in feats if feat.GetFamily() in ["Aromatic_1"]]

    G = nx.Graph()
    atom_ids = []
    for n, (feat_name, feat_pos, feat_atomid) in enumerate(feats_pharma, 0):
        atom_ids.append(feat_atomid)
        if feat_name == "Hydrophilic":
            G.add_node(n, define=f"{feat_name}")
        else:
            num_nos = sum([mol.GetAtomWithIdx(i).GetSymbol() in ["N", "O", "S"] for i in feat_atomid])
            if feat_pos in feats_aromatic:
                G.add_node(n, define=f"{feat_name}_aro1_{num_nos}")
            else:
                G.add_node(n, define=f"{feat_name}_aro0_{num_nos}")

    feat_len = len(feats_pharma)
    m = np.zeros([feat_len, feat_len])
    for i in range(feat_len):
        for j in range(feat_len):
            m[i, j] = euclidist(feats_pharma[i][1], feats_pharma[j][1])

    max_m = np.max(m[0])

    bond = find_connect(atom_ids, mol)
    for begin, end in bond:
        bin = find_bin(m[begin, end], distance_bin)
        G.add_edge(begin, end, distance=bin)
    return G, max_m


def collect_partial_frag_graph_2(smile, vocab, collect_type="component", enhance=False):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    mol = Chem.RemoveHs(mol)
    ori_smi = Chem.MolToSmiles(mol)
    AllChem.Compute2DCoords(mol)
    feats = factory.GetFeaturesForMol(mol)
    feats_pharma = [(feat.GetFamily(), get_point(feat.GetPos()), feat.GetAtomIds()) for feat in feats if
                    feat.GetFamily() not in ["Aromatic_1", "Aromatic_2"]]
    feats_pharma = np.array(feats_pharma, dtype=object)
    # feats_aromatic = [get_point(feat.GetPos()) for feat in feats if feat.GetFamily() in ["Aromatic_1"]]

    feat_len = len(feats_pharma)
    m = np.zeros([feat_len, feat_len])
    # len为1的情况 不具备参考价值
    for i in range(feat_len):
        for j in range(feat_len):
            m[i, j] = euclidist(feats_pharma[i][1], feats_pharma[j][1])

    collect_smi = set()   # 有问题
    for diameter, vocab_list in vocab.items():
        for dis in m:
            idx = np.where(dis <= diameter)[0]
            cur_feats_pharma = feats_pharma[idx]

            atom_ids = []
            for n, (feat_name, feat_pos, feat_atomid) in enumerate(cur_feats_pharma, 0):
                atom_ids.append(feat_atomid)
            frag_smi = partial_collect_for_subgraph(mol, atom_ids)
            if frag_smi is not None:
                for s in frag_smi:
                    graph, _ = create_frag_graph(Chem.MolFromSmiles(s))
                    if collect_type == "topo&distance":
                        hash = nx.weisfeiler_lehman_graph_hash(graph, edge_attr="distance", node_attr="define")
                        if hash in vocab_list:
                            collect_smi.add(f"{s},{hash},{ori_smi}")

                    if collect_type == "topo":
                        hash = nx.weisfeiler_lehman_graph_hash(graph, node_attr="define")
                        if hash in vocab_list:
                            collect_smi.add(f"{s},{hash},{ori_smi}")

            if enhance is True:
                for i in range(len(cur_feats_pharma)):  # 尝试减少一个特征增加匹配概率
                    cur_pharma = cur_feats_pharma.copy().tolist()
                    cur_pharma.pop(i)
                    atom_ids = []
                    for n, (feat_name, feat_pos, feat_atomid) in enumerate(cur_pharma, 0):
                        atom_ids.append(feat_atomid)
                    frag_smi = partial_collect_for_subgraph(mol, atom_ids)
                if frag_smi is not None:
                    for s in frag_smi:
                        graph, _ = create_frag_graph(Chem.MolFromSmiles(s))
                        if collect_type == "topo&distance":
                            hash = nx.weisfeiler_lehman_graph_hash(graph, edge_attr="distance", node_attr="define")
                            if hash in vocab_list:
                                collect_smi.add(f"{s},{hash},{ori_smi}")
                        if collect_type == "topo":
                            hash = nx.weisfeiler_lehman_graph_hash(graph, node_attr="define")
                            if hash in vocab_list:
                                collect_smi.add(f"{s},{hash},{ori_smi}")
    return collect_smi



if __name__ == '__main__':
    mol = Chem.MolFromSmiles("Cc(n1)cc(C)c(c12)CN(C2)C(=O)CC3CN(C3)c4ccnc(C(F)(F)F)c4")


