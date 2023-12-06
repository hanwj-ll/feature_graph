import pandas as pd
import mdtraj as md
from rdkit import Chem
from tqdm import tqdm
from help.pocket_split import partial_collect, remain
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')


def load_rec(path="IRG1.pdb"):
    t = md.load_pdb(path)
    cor = t.xyz * 10
    top = t.topology
    cor_re = pd.DataFrame(cor.reshape(len(cor[0]), 3), columns=['x', 'y', 'z'])
    table, _ = top.to_dataframe()
    merge = pd.concat([table, cor_re], axis=1)
    pocket = merge[(merge['resSeq'] == 158) & (merge['name'] == 'O')|
                   (merge['resSeq'] == 246) & (merge['name'] == 'CE2')]
    return pocket


if __name__ == '__main__':
    count1 = 0
    count2 = 0
    poc = load_rec("data/IRG1.pdb")
    suppl = Chem.SDMolSupplier("data/candidate_compounds.sdf", removeHs=True)
    writer = Chem.SDWriter('data/seed_fragment.sdf')
    writer2 = Chem.SDWriter('data/other.sdf')
    for mol in tqdm(suppl, desc="process mol "):
        if mol is None:
            continue
        frag, smi, frag_id = partial_collect(mol, poc)
        if frag is not None:
            ori_smi = Chem.MolToSmiles(mol)
            frag.SetProp('ori_smi', f'{ori_smi}')
            frag.SetProp('smi', f'{smi}')
            try:
                writer.write(frag)
                count1 += 1
            except Exception as e:
                print(e)
            rem, smi_rem = remain(mol, frag_id)
            if rem is not None:
                rem.SetProp('ori_smi', f'{ori_smi}')
                rem.SetProp('smi', f'{smi_rem}')
                try:
                    writer2.write(rem)
                    count2 += 1
                except Exception as e:
                    print(e)

    print(f"收集{count1} 个seed片段, {count2}个other片段")
    writer.close()
    writer2.close()
