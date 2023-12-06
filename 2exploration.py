from functools import partial
import pandas as pd
from help.dis_pharma import collect_partial_frag_graph_2
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
import sys
sys.path.append("/public/home/cadd1/lander/pytmp")

def search_pharma(lines, vocab, type="topo&distance"):
    import traceback
    result = set()
    for line in lines:
        smile = line.split(" ")[0]
        try:
            res = collect_partial_frag_graph_2(smile, vocab, collect_type=type, enhance=False)
            if res is None:
                continue
            result.update(res)
        except:
            print('--------traceback.format_exc--------')
            error =traceback.format_exc()
            print(error)
            raise Exception(error)
    return result

def search_mol(num_cpu, vocab, col_type="topo&distance", path="data/validation.csv"):
    from multiprocessing import Pool, cpu_count
    ncpu = min(cpu_count(), max(num_cpu, 1))
    p = Pool(ncpu)
    # suppl = Chem.SmilesMolSupplier('data/zinc_druglike.smi', smilesColumn=0, titleLine=False, nameColumn=False)
    with open('data/chembl31_druglike.smi', 'rt') as f:
        mols_text = f.readlines()
    batch_size = len(mols_text) // ncpu + 1
    batches = [mols_text[i: i + batch_size] for i in range(0, len(mols_text), batch_size)]
    # result = dict()
    result = set()
    for i, res in enumerate(p.imap_unordered(partial(search_pharma, vocab=vocab, type=col_type), batches), 1):
        result.update(res)
        print(f'\r {i} batch collected,  collect {len(res)} frags, total {len(result)} frags ')
    print()
    print(f"collect {len(result)} frags")
    with open(path, 'wt') as f:
        f.write('\n'.join([line for line in result]))
    return 0

def vocab_diameter(df):

    vocab_dia = dict()
    for index, row in df.iterrows():
        fea = row.hash_distance
        dia = row.diameter  # math.ceil()
        if dia not in vocab_dia.keys():
            vocab_dia[dia] = set()
        vocab_dia[dia].add(fea)

    res = dict()
    for key, value in vocab_dia.items():
        res[key] = list(value)
    return res


if __name__ == '__main__':

    df = pd.read_csv("data/seed_fragment.csv")
    print("Searching under topo&distance constraints")
    vocab = vocab_diameter(df)
    a = search_mol(60, vocab, col_type="topo&distance", path="data/search_chembl.csv")

