import multiprocessing
from multiprocessing import Pool
import io
import os
import itertools
import re
import networkx as nx
import pydock
from func_timeout import func_set_timeout
from tqdm import tqdm
from src.data import algos
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Conformer
from rdkit import RDLogger
from torch.autograd import Variable
from src.utils.dist_to_coords_utils import get_mask_rotate, modify_conformer, modify_conformer2, axis_angle_to_matrix

RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import rdMolTransforms
import copy
import lmdb
import pickle
import pandas as pd
import torch
import torch.nn.functional as F
from multiprocessing import Pool
from tqdm import tqdm
import glob
from scipy.optimize import differential_evolution
from torch.distributions import Normal
from spyrmsd import rmsd, molecule, graph
from lion_pytorch import Lion
import prody
from spyrmsd import rmsd, molecule
# from func_timeout import func_set_timeout
from pathlib import Path





def get_torsions(m, removeHs=True):
    if removeHs:
        m = Chem.RemoveHs(m)
    torsionList = []
    torsionSmarts = "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"
    torsionQuery = Chem.MolFromSmarts(torsionSmarts)
    matches = m.GetSubstructMatches(torsionQuery)
    for match in matches:
        idx2 = match[0]
        idx3 = match[1]
        bond = m.GetBondBetweenAtoms(idx2, idx3)
        jAtom = m.GetAtomWithIdx(idx2)
        kAtom = m.GetAtomWithIdx(idx3)
        for b1 in jAtom.GetBonds():
            if b1.GetIdx() == bond.GetIdx():
                continue
            idx1 = b1.GetOtherAtomIdx(idx2)
            for b2 in kAtom.GetBonds():
                if (b2.GetIdx() == bond.GetIdx()) or (b2.GetIdx() == b1.GetIdx()):
                    continue
                idx4 = b2.GetOtherAtomIdx(idx3)
                # skip 3-membered rings
                if idx4 == idx1:
                    continue
                # skip torsions that include hydrogens
                if (m.GetAtomWithIdx(idx1).GetAtomicNum() == 1) or (
                        m.GetAtomWithIdx(idx4).GetAtomicNum() == 1
                ):
                    continue
                if m.GetAtomWithIdx(idx4).IsInRing():
                    torsionList.append((idx4, idx3, idx2, idx1))
                    break
                else:
                    torsionList.append((idx1, idx2, idx3, idx4))
                    break
            break
    return torsionList


def SetDihedral(conf, atom_idx, new_vale):
    rdMolTransforms.SetDihedralRad(
        conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale
    )


def GetDihedral(conf, atom_idx):
    return rdMolTransforms.GetDihedralRad(
        conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3]
    )


def single_conf_gen_bonds(tgt_mol, num_confs=1000, seed=42, removeHs=True):
    mol = copy.deepcopy(tgt_mol)
    mol = Chem.AddHs(mol)
    allconformers = AllChem.EmbedMultipleConfs(
        mol, numConfs=num_confs, randomSeed=seed, clearConfs=True
    )
    if removeHs:
        mol = Chem.RemoveHs(mol)
    rotable_bonds = get_torsions(mol, removeHs=removeHs)
    for i in range(len(allconformers)):
        np.random.seed(i)
        values = 3.1415926 * 2 * np.random.rand(len(rotable_bonds))
        for idx in range(len(rotable_bonds)):
            SetDihedral(mol.GetConformers()[i], rotable_bonds[idx], values[idx])
        Chem.rdMolTransforms.CanonicalizeConformer(mol.GetConformers()[i])
    return mol


def load_lmdb_data(lmdb_path, key):
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    txn = env.begin()
    _keys = list(txn.cursor().iternext(values=False))
    collects = dict()
    for idx in range(len(_keys)):
        datapoint_pickled = txn.get(f"{idx}".encode("ascii"))
        data = pickle.loads(datapoint_pickled)
        # collects.append(data[key])
        collects[data['pocket']] = [Chem.RemoveHs(mol) for mol in data[key]]
    return collects


def docking_data_pre(raw_data_path, predict):
    mol_dict = load_lmdb_data(os.path.join(raw_data_path, 'test.lmdb'), "mol_list")
    # mol_list = [Chem.RemoveHs(mol) for items in mol_list for mol in items]
    (
        smi_list,
        pocket_list,
        pocket_coords_list,
        distance_predict_list,
        holo_distance_predict_list,
        holo_coords_list,
        holo_center_coords_list,
        pis,
        mus,
        sigmas,
        init_coords_list,
    ) = ([], [], [], [], [], [], [], [], [], [], [])
    for batch in predict:
        sz = batch["atoms"].size(0)

        for i in range(sz):
            smi_list.append(batch["smi_name"][i])
            pocket_list.append(batch["pocket_name"][i])

            distance_predict = batch["cross_distance_predict"][i]
            token_mask = batch["atoms"][i] > 2
            pocket_token_mask = batch["pocket_atoms"][i] > 2
            distance_predict = distance_predict[token_mask][:, pocket_token_mask]
            pocket_coords = batch["pocket_coordinates"][i]
            pocket_coords = pocket_coords[pocket_token_mask, :]
            pi = batch['pi'][i][token_mask][:, pocket_token_mask]
            mu = batch['mu'][i][token_mask][:, pocket_token_mask]
            sigma = batch['sigma'][i][token_mask][:, pocket_token_mask]

            holo_distance_predict = batch["holo_distance_predict"][i]
            holo_distance_predict = holo_distance_predict[token_mask][:, token_mask]

            holo_coordinates = batch["holo_coordinates"][i]
            holo_coordinates = holo_coordinates[token_mask, :]
            holo_center_coordinates = batch["holo_center_coordinates"][i][:3]

            pocket_coords = pocket_coords.numpy().astype(np.float32)
            distance_predict = distance_predict.numpy().astype(np.float32)
            holo_distance_predict = holo_distance_predict.numpy().astype(np.float32)
            holo_coords = holo_coordinates.numpy().astype(np.float32)

            pocket_coords_list.append(pocket_coords)
            distance_predict_list.append(distance_predict)
            holo_distance_predict_list.append(holo_distance_predict)
            holo_coords_list.append(holo_coords)
            holo_center_coords_list.append(holo_center_coordinates)

            init_coords_list.append(batch['holo_init_coordinates'][i][token_mask, :])

            pis.append(pi)
            mus.append(mu)
            sigmas.append(sigma)

    return (
        mol_dict,
        smi_list,
        pocket_list,
        pocket_coords_list,
        distance_predict_list,
        holo_distance_predict_list,
        holo_coords_list,
        holo_center_coords_list,
        pis,
        mus,
        sigmas,
        init_coords_list
    )


def ensemble_iterations(
        mol_dict,
        smi_list,
        pocket_list,
        pocket_coords_list,
        distance_predict_list,
        holo_distance_predict_list,
        holo_coords_list,
        holo_center_coords_list,
        pi_list,
        mu_list,
        sigma_list,
        init_coords_list,
        tta_times=10,
        sample_times=10,
):
    sz = len(pocket_list)
    # hash_mol_list = dict()
    # smi_set = set()
    # for i in range(sz // tta_times):
    #     start_idx, end_idx = i * tta_times, (i + 1) * tta_times
    #     smi = Chem.MolToSmiles(mol_list[start_idx])
    #     hash_mol_list[smi] = start_idx
    #     smi_set.add(smi)
    # smi_list = list(smi_set)
    # assert len(hash_mol_list) == (sz // tta_times)

    for i in range(sz // tta_times):
        # indices = [j for j, smi in enumerate(smi_list) if smi == smi_list[i]]
        start_idx, end_idx = i * tta_times, (i + 1) * tta_times
        distance_predict_tta = distance_predict_list[start_idx:end_idx]
        # distance_predict_tta = [distance_predict_list[j] for j in indices]
        holo_distance_predict_tta = holo_distance_predict_list[start_idx:end_idx]
        pi_tta = pi_list[start_idx:end_idx]
        mu_tta = mu_list[start_idx:end_idx]
        sigma_tta = sigma_list[start_idx:end_idx]

        # distance_predict_tta = [(torch.softmax(pi, dim=-1) * mu).sum(dim=-1).cpu().data.numpy() for pi, mu in
        #                         zip(pi_tta, mu_tta)]

        pi = torch.softmax(torch.stack(pi_tta).mean(dim=0), dim=-1)
        mu = torch.stack(mu_tta).mean(dim=0)
        sigma = torch.stack(sigma_tta).mean(dim=0)

        # mol_index = hash_mol_list.get(smi_list[start_idx])
        # mol_index = start_idx
        # mol = copy.deepcopy(mol_dict[mol_index])
        mol = copy.deepcopy(mol_dict[pocket_list[start_idx]][0])
        rdkit_mol = single_conf_gen_bonds(
            mol, num_confs=sample_times, seed=42, removeHs=True
        )
        sz = len(rdkit_mol.GetConformers())
        initial_coords_list = [
            rdkit_mol.GetConformers()[i].GetPositions().astype(np.float32)
            for i in range(sz)
        ]
        # initial_coords_list = init_coords_list[start_idx:end_idx][:sample_times]

        yield [
            initial_coords_list,
            mol,
            smi_list[start_idx],
            pocket_list[start_idx],
            pocket_coords_list[start_idx],
            distance_predict_tta,
            holo_distance_predict_tta,
            holo_coords_list[start_idx],
            holo_center_coords_list[start_idx],
            pi,
            mu,
            sigma,
        ]


def rmsd_func(holo_coords, predict_coords):
    if predict_coords is not np.nan:
        sz = holo_coords.shape
        # rmsd = np.sqrt(np.sum((predict_coords - holo_coords) ** 2) / sz[0])
        rmsd = np.sqrt(np.max(np.sum((predict_coords - holo_coords) ** 2, axis=-1)))
        return rmsd
    return 1000.0


def print_results(rmsd_results):
    print("RMSD < 0.5 : ", np.mean(rmsd_results < 0.5))
    print("RMSD < 1.0 : ", np.mean(rmsd_results < 1.0))
    print("RMSD < 1.5 : ", np.mean(rmsd_results < 1.5))
    print("RMSD < 2.0 : ", np.mean(rmsd_results < 2.0))
    print("RMSD < 2.5 : ", np.mean(rmsd_results < 2.5))
    print("RMSD < 3.0 : ", np.mean(rmsd_results < 3.0))
    print("RMSD < 4.0 : ", np.mean(rmsd_results < 4.0))
    print("RMSD < 5.0 : ", np.mean(rmsd_results < 5.0))
    print("avg RMSD : ", np.mean(rmsd_results))


def single_SF_loss(
        predict_coords,
        pocket_coords,
        distance_predict,
        holo_distance_predict,
        dist_threshold=6,
        holo_dist_cons=None,
        distance_mask=None
):
    dist = torch.norm(predict_coords.unsqueeze(1) - pocket_coords.unsqueeze(0), dim=-1)
    holo_dist = torch.norm(
        predict_coords.unsqueeze(1) - predict_coords.unsqueeze(0), dim=-1
    )
    if distance_mask is None:
        distance_mask = distance_predict < dist_threshold
        # distance_mask = get_k_nearest_mask(distance_predict, 8)
    cross_dist_score = F.smooth_l1_loss(distance_predict[distance_mask], dist[distance_mask])
    dist_score = F.smooth_l1_loss(holo_distance_predict, holo_dist)
    loss = cross_dist_score * 1.0 + dist_score * 1.0
    return loss


def loss_with_isomorphisms(
        predict_coords,
        pocket_coords,
        distance_predict,
        holo_distance_predict,
        dist_threshold=6,
        isomorphisms=None,
        distance_mask=None,

):
    loss = torch.inf
    for iso in isomorphisms:
        predict_coords = predict_coords[iso, :]
        dist = torch.norm(predict_coords.unsqueeze(1) - pocket_coords.unsqueeze(0), dim=-1)
        holo_dist = torch.norm(
            predict_coords.unsqueeze(1) - predict_coords.unsqueeze(0), dim=-1
        )
        if distance_mask is None:
            distance_mask = distance_predict < dist_threshold
            # distance_mask = get_k_nearest_mask(distance_predict, 8)
        cross_dist_score = F.smooth_l1_loss(distance_predict[distance_mask], dist[distance_mask])
        dist_score = F.smooth_l1_loss(holo_distance_predict, holo_dist)
        tmp_loss = cross_dist_score * 1.0 + dist_score * 1.0
        if tmp_loss < loss:
            loss = tmp_loss
    return loss


def match_aprops(node1, node2):
    """
    Check if atomic properties for two nodes match.
    """
    return node1["aprops"] == node2["aprops"]


def _handle_timeout(signum, frame):
    raise TimeoutError('function timeout')


@func_set_timeout(10)
def get_isomorphisms(rdkit_mol, max_nums=100):
    # Define the molecule

    # Create the molecule object
    mol = molecule.Molecule.from_rdkit(rdkit_mol)
    g1 = graph.graph_from_adjacency_matrix(mol.adjacency_matrix, mol.atomicnums)
    gm = nx.algorithms.isomorphism.GraphMatcher(g1, g1, match_aprops)
    isomorphisms = [[iso[i] for i in range(len(g1.nodes))] for iso in gm.isomorphisms_iter()]
    # Get all the possible graph isomorphisms
    # isomorphisms = graph.match_graphs(g1, g2)
    return isomorphisms[:max_nums]


def holo_loss(predict_coords, holo_distance_predict, mask=None):
    if mask is None:
        mask = torch.ones_like(holo_distance_predict, dtype=torch.bool)
    dist = (predict_coords.unsqueeze(1) - predict_coords.unsqueeze(0)).norm(dim=-1)
    loss = F.smooth_l1_loss(dist[mask], holo_distance_predict[mask])
    return loss


def single_docking_loss(
        predict_coords,
        pocket_coords,
        distance_predict,
        dist_threshold=6,
):
    dist = torch.norm(predict_coords.unsqueeze(1) - pocket_coords.unsqueeze(0), dim=-1)
    distance_mask = distance_predict < dist_threshold
    # cross_dist_score = (
    #         (dist[distance_mask] - distance_predict[distance_mask]) ** 2
    # ).mean()
    # dist_score = ((holo_dist - holo_distance_predict) ** 2).mean()
    cross_dist_score = F.smooth_l1_loss(distance_predict[distance_mask], dist[distance_mask])
    loss = cross_dist_score
    return loss


def scoring(
        predict_coords,
        pocket_coords,
        distance_predict,
        holo_distance_predict,
        dist_threshold=4.5,
):
    predict_coords = predict_coords.detach()
    dist = torch.norm(predict_coords.unsqueeze(1) - pocket_coords.unsqueeze(0), dim=-1)
    holo_dist = torch.norm(
        predict_coords.unsqueeze(1) - predict_coords.unsqueeze(0), dim=-1
    )
    distance_mask = distance_predict < dist_threshold
    cross_dist_score = (
            (dist[distance_mask] - distance_predict[distance_mask]) ** 2
    ).mean()
    dist_score = ((holo_dist - holo_distance_predict) ** 2).mean()
    return cross_dist_score.numpy(), dist_score.numpy()


def dock_with_gradient(
        coords,
        pocket_coords,
        distance_predict_tta,
        holo_distance_predict_tta,
        pi=None,
        mu=None,
        sigma=None,
        loss_func=single_SF_loss,
        holo_coords=None,
        iterations=20000,
        early_stoping=5,
        holo_dist_cons=None,
):
    bst_loss, bst_coords, bst_meta_info = 10000.0, coords, None
    all_coords = []
    for i, (distance_predict, holo_distance_predict) in enumerate(
            zip(distance_predict_tta, holo_distance_predict_tta)
    ):
        new_coords = copy.deepcopy(coords)
        _coords, _loss, _meta_info = single_dock_with_gradient(
            coords=torch.from_numpy(new_coords).float(),
            pocket_coords=torch.from_numpy(pocket_coords).float(),
            distance_predict=torch.from_numpy(distance_predict).float(),
            holo_distance_predict=torch.from_numpy(holo_distance_predict).float(),
            pi=pi,
            mu=mu,
            sigma=sigma,
            loss_func=loss_func,
            holo_coords=holo_coords,
            iterations=iterations,
            early_stoping=early_stoping,
            holo_dist_cons=holo_dist_cons,
            distance_predict_tta=[torch.from_numpy(d) for d in distance_predict_tta],
            holo_distance_predict_tta=[torch.from_numpy(d) for d in holo_distance_predict_tta],
        )
        # if bst_loss > _loss:
        #     bst_coords = _coords
        #     bst_loss = _loss
        #     bst_meta_info = _meta_info
        all_coords.append((_coords, _loss, _meta_info))
    return all_coords


def single_dock_with_gradient(
        coords,
        pocket_coords,
        distance_predict,
        holo_distance_predict,
        pi=None,
        mu=None,
        sigma=None,
        loss_func=single_SF_loss,
        holo_coords=None,
        iterations=20000,
        early_stoping=5,
        holo_dist_cons=None,
        distance_predict_tta=None,
        holo_distance_predict_tta=None,
):
    if holo_coords is not None:
        holo_coords = torch.from_numpy(holo_coords).float()

    coords.requires_grad = True
    optimizer = torch.optim.LBFGS([coords], lr=0.1)
    bst_loss, times, best_coords = 10000.0, 0, None
    for i in range(iterations):

        def closure():
            optimizer.zero_grad()
            loss = loss_func(
                coords, pocket_coords, distance_predict, holo_distance_predict, holo_dist_cons=holo_dist_cons
            )
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        if loss.item() < bst_loss:
            bst_loss = loss.item()
            times = 0
            best_coords = copy.deepcopy(coords).detach()
        else:
            times += 1
            if times > early_stoping:
                break
    score = mdn_score(pi, mu, sigma, best_coords, pocket_coords)
    _loss = np.min([single_SF_loss(best_coords, pocket_coords, d1, d2).item() for d1, d2 in
                    zip(distance_predict_tta, holo_distance_predict_tta)])
    return best_coords.cpu().numpy(), _loss.item(), score.item()


def optimize_coords(
        coords,
        holo_distance_predict,
        loss_func=holo_loss,
        iterations=10000,
        early_stoping=5,
        mask=None
):
    coords.requires_grad = True
    optimizer = torch.optim.LBFGS([coords], lr=0.1)
    best_loss, times, best_coords = 10000.0, 0, None
    for i in range(iterations):

        def closure():
            optimizer.zero_grad()
            loss = loss_func(coords, holo_distance_predict, mask)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        if loss.item() < best_loss:
            best_loss = loss.item()
            times = 0
            best_coords = copy.deepcopy(coords).detach()
        else:
            times += 1
            if times > early_stoping:
                break
    if best_loss < 1.0:
        best_coords.requires_grad = False
        return best_coords

# ls revise (this one for train)
def set_coord_ori(mol, coords):
    _mol = copy.deepcopy(mol)
    for i in range(coords.shape[0]):
        _mol.GetConformer().SetAtomPosition(i, coords[i].tolist())
    return _mol
##this one for infer
def set_coord(mol, coords, idx=0):
    _mol = copy.deepcopy(mol)
    if len(_mol.GetConformers()) == 0:
        conf = Conformer(len(_mol.GetAtoms()))
        for i in range(len(_mol.GetAtoms())):
            conf.SetAtomPosition(i, coords[i].tolist())
        _mol.AddConformer(conf)
    else:
        for i in range(coords.shape[0]):
            _mol.GetConformer(idx).SetAtomPosition(i, coords[i].tolist())
    return _mol

def add_coord(mol, xyz):
    x, y, z = xyz
    conf = mol.GetConformer()
    pos = conf.GetPositions()
    pos[:, 0] += x
    pos[:, 1] += y
    pos[:, 2] += z
    for i in range(pos.shape[0]):
        conf.SetAtomPosition(
            i, Chem.rdGeometry.Point3D(pos[i][0], pos[i][1], pos[i][2])
        )
    return mol


def get_n_hop_matrix(mol):
    edges_list = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # add edges in both directions
        edges_list.append((i, j))
        edges_list.append((j, i))
    edge_index = np.array(edges_list).T
    n = len(mol.GetAtoms())
    adj = torch.zeros([n, n], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True
    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    return torch.from_numpy(shortest_path_result)


def multithreading_process_single_docking(input_path, output_path, output_ligand_path):
    content = pd.read_pickle(input_path)
    (
        init_coords_tta,
        mol,
        smi,
        pocket,
        pocket_coords,
        distance_predict_tta,
        holo_distance_predict_tta,
        holo_coords,
        holo_center_coords,
        pi,
        mu,
        sigma,
    ) = content
    sample_times = len(init_coords_tta)
    holo_coords = mol.GetConformer().GetPositions() - holo_center_coords.unsqueeze(0).cpu().data.numpy()
    # distance_predict_tta = distance_predict_tta[:5]
    # holo_distance_predict_tta = holo_distance_predict_tta[:5]
    # pt_holo_coords = torch.from_numpy(holo_coords)
    # pt_pocket_coords = torch.from_numpy(pocket_coords)
    # distance_predict_tta = [(pt_holo_coords.unsqueeze(1)-pt_pocket_coords.unsqueeze(0)).norm(dim=-1).cpu().data.numpy()]
    # holo_distance_predict_tta = [(pt_holo_coords.unsqueeze(1)-pt_holo_coords.unsqueeze(0)).norm(dim=-1).cpu().data.numpy()]

    # modified_mol_list = dist2coords(init_coords_tta, mol, pocket_coords, distance_predict_tta,
    #                                 holo_distance_predict_tta, sample_times, holo_coords, pi, mu, sigma)
    # modified_mol_list = dist2coords_with_de(init_coords_tta, mol, pocket_coords, distance_predict_tta,
    #                                         holo_distance_predict_tta, sample_times, holo_coords, pi, mu, sigma)
    modified_mol_list = dist_to_coords_with_tor(init_coords_tta, mol, pocket_coords, distance_predict_tta,
                                                holo_distance_predict_tta, sample_times, holo_coords, pi, mu, sigma)
    # modified_mol_list = dist_to_coords_with_cuda(init_coords_tta, mol, pocket_coords, distance_predict_tta,
    #                                              holo_distance_predict_tta, sample_times, holo_coords, pi, mu, sigma)

    log_data, modified_mol_list = prepare_log_data(modified_mol_list, pocket_coords, distance_predict_tta,
                                                   holo_distance_predict_tta, pi, mu, sigma)

    try:
        # _rmsd = [round(rmsd_func(holo_coords, item[0]), 4) for item in log_data]
        _rmsd = [round(i, 4) for i in get_symmetry_rmsd(mol, holo_coords, [item[0] for item in log_data])]
        print(f"{pocket}-{smi}-TOP1/5/10/all_RMSD:{_rmsd[0]}-{min(_rmsd[:5])}-{min(_rmsd[:10])}-{min(_rmsd)}")
    except:
        print(input_path)

    if output_path is not None:
        with open(output_path, "wb") as f:
            pickle.dump(
                [log_data, holo_coords, smi, pocket, pocket_coords, mol],
                f,
            )
    if output_ligand_path is not None:
        modified_mol_list = [add_coord(mol, holo_center_coords.numpy()) for mol in modified_mol_list]
        save_sdf(modified_mol_list, output_ligand_path)

    return True


def dist2coords(init_coords_tta, mol, pocket_coords, distance_predict_tta, holo_distance_predict_tta, sample_times,
                holo_coords, pi, mu, sigma):
    all_predict_coords = []
    for i in range(sample_times):
        init_coords = init_coords_tta[i]

        # holo_dist_cons = (torch.from_numpy(init_coords).unsqueeze(1) - torch.from_numpy(init_coords).unsqueeze(0)).norm(
        #     dim=-1)
        # holo_dist_cons[n_hop_mask > 2] = 0
        n_hop_mask = get_n_hop_matrix(mol)
        # holo_dist_cons = None

        predict_coords = dock_with_gradient(
            init_coords,
            pocket_coords,
            distance_predict_tta,
            holo_distance_predict_tta,
            pi=pi,
            mu=mu,
            sigma=sigma,
            holo_coords=holo_coords,
            loss_func=single_SF_loss,
            holo_dist_cons=n_hop_mask,
        )
        # if loss < bst_loss:
        #     bst_loss = loss
        #     bst_predict_coords = predict_coords
        #     bst_meta_info = meta_info
        all_predict_coords.append(predict_coords)

    # 从多个sample中选取拟合得最好的坐标
    selected_coords = []
    for j in range(len(all_predict_coords[0])):
        for i in range(sample_times):
            if all_predict_coords[i][j][1] < 100:
                selected_coords.append((all_predict_coords[i][j], init_coords_tta[i]))

    sorted_data = sorted(selected_coords, key=lambda x: x[0][2] - 40 * x[0][1], reverse=True)

    mol = Chem.RemoveHs(mol)
    mol_list = [set_coord(mol, pred_coords[0][0]) for pred_coords in sorted_data]
    # modified_mol_list = [set_coord(mol, init_coords[1]) for init_coords in sorted_data]
    # rotable_bonds = get_torsions(mol)
    # if len(rotable_bonds) == 0:
    #     for rdkit_mol, mol in zip(modified_mol_list, mol_list):
    #         AllChem.AlignMol(rdkit_mol, mol)
    # else:
    #     modified_mol_list = [optimize_rotatable_bonds(rdkit_mol, mol, rotable_bonds) for rdkit_mol, mol in
    #                          zip(modified_mol_list, mol_list)]
    modified_mol_list = mol_list
    return modified_mol_list


def dist2coords_with_de(init_coords_tta, mol, pocket_coords, distance_predict_tta,
                        holo_distance_predict_tta, *args, **kwargs):
    mol = Chem.RemoveHs(mol)
    init_mol_list = [set_coord(mol, init_coords - init_coords.mean(axis=0)) for init_coords in init_coords_tta]
    rotable_bonds = get_torsions(mol)
    modified_mol_list = []
    pt_pocket_coords = torch.from_numpy(pocket_coords)
    for init_mol in init_mol_list:
        for cross_dist, holo_dist in zip(distance_predict_tta, holo_distance_predict_tta):
            _mol = optimize_conformer(copy.deepcopy(init_mol), rotable_bonds, pt_pocket_coords, cross_dist, holo_dist)
            modified_mol_list.append(_mol)
    return modified_mol_list


def dist_to_coords_with_cuda(init_coords_tta, mol, pocket_coords, distance_predict_tta,
                             holo_distance_predict_tta, sample_times, holo_coords, pi, mu, sigma, iterations=20000,
                             early_stoping=5):
    pt_pocket_coords = torch.from_numpy(pocket_coords)
    torsions, masks = get_mask_rotate(mol)
    pred_coords = []
    pt_distance_predict_tta = [torch.from_numpy(d) for d in distance_predict_tta]
    pt_holo_distance_predict_tta = [torch.from_numpy(d) for d in holo_distance_predict_tta]
    for init_coord in init_coords_tta:
        init_coord = torch.from_numpy(init_coord - init_coord.mean(axis=0))
        for pred_cross_dist, pred_holo_dist in zip(pt_distance_predict_tta, pt_holo_distance_predict_tta):
            values = Variable(torch.zeros(6 + len(torsions)), requires_grad=True)
            best_values, best_loss, ok = pydock.lbfgsb(init_coord, torsions, masks, pt_pocket_coords,
                                                       pred_cross_dist, pred_holo_dist, values, eps=0.01)

            if best_loss < 100:
                best_coords = modify_conformer(init_coord, best_values, torsions, masks)
                best_score = mdn_score(pi, mu, sigma, best_coords, pt_pocket_coords).item()
                pred_coords.append((best_coords.cpu().data.numpy(), best_loss, best_score))
    sorted_data = sorted(pred_coords, key=lambda x: 5 * x[1] - x[2])
    modified_mol_list = [set_coord(mol, coords[0]) for coords in sorted_data]
    return modified_mol_list


def dist_to_coords_with_tor(init_coords_tta, mol, pocket_coords, distance_predict_tta,
                            holo_distance_predict_tta, sample_times, holo_coords, pi, mu, sigma, iterations=20000,
                            early_stoping=5):
    pt_pocket_coords = torch.from_numpy(pocket_coords)
    torsions, masks = get_mask_rotate(mol)
    # isomorphisms = get_isomorphisms(mol)
    pred_coords = []
    # hop_matrix = get_n_hop_matrix(mol)
    # holo_mask = (0 < hop_matrix) & (hop_matrix < 3)
    pt_distance_predict_tta = [torch.from_numpy(d) for d in distance_predict_tta]
    pt_holo_distance_predict_tta = [torch.from_numpy(d) for d in holo_distance_predict_tta]
    for init_coord in init_coords_tta:
        init_coord = torch.from_numpy(init_coord - init_coord.mean(axis=0))
        for pred_cross_dist, pred_holo_dist in zip(pt_distance_predict_tta, pt_holo_distance_predict_tta):
            # opt_coord = optimize_coords(copy.deepcopy(init_coord), pred_holo_dist, mask=holo_mask)
            # if opt_coord is not None:
            #     init_coord = opt_coord
            #     init_coord = init_coord - init_coord.mean(dim=0)
            # distance_mask = get_k_nearest_mask(pred_cross_dist, 8)
            distance_mask = pred_cross_dist < 6
            # values = Variable(torch.cat([torch.zeros(3), torch.rand(6 + 2 * len(torsions))]), requires_grad=True)
            values = Variable(torch.zeros(6 + len(torsions)), requires_grad=True)
            # values = Variable(torch.zeros(6), requires_grad=True)
            optimizer = torch.optim.LBFGS([values], lr=0.1)
            best_loss, times, best_values, best_score, best_coords = 10000.0, 0, None, 0, None
            for i in range(iterations):

                def closure():
                    optimizer.zero_grad()
                    new_pos = modify_conformer(init_coord, values, torsions, masks)
                    loss = single_SF_loss(new_pos, pt_pocket_coords, pred_cross_dist, pred_holo_dist,
                                          distance_mask=distance_mask)
                    # loss = loss_with_isomorphisms(new_pos, pt_pocket_coords, pred_cross_dist, pred_holo_dist,
                    #                               isomorphisms=isomorphisms,
                    #                               distance_mask=distance_mask)
                    loss.backward()
                    return loss

                # loss = closure()
                # optimizer.step()
                loss = optimizer.step(closure)
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    times = 0
                    best_values = copy.deepcopy(values).detach()
                else:
                    times += 1
                    if times > early_stoping:
                        break
            if best_loss < 100:
                best_coords = modify_conformer(init_coord, best_values, torsions, masks)
                best_score = mdn_score(pi, mu, sigma, best_coords, pt_pocket_coords).item()
                # best_loss = single_SF_loss(best_coords, pt_pocket_coords, pred_cross_dist, pred_holo_dist,
                #                            distance_mask=distance_mask).item()
                pred_coords.append((best_coords.cpu().data.numpy(), best_loss, best_score))
    sorted_data = sorted(pred_coords, key=lambda x: 5 * x[1] - x[2])
    modified_mol_list = [set_coord(mol, coords[0]) for coords in sorted_data]
    return modified_mol_list


def mdn_score(pi, mu, sigma, predict_coords, pocket_coords, threshold=5):
    dist = torch.norm(predict_coords.unsqueeze(1) - pocket_coords.unsqueeze(0), dim=-1)
    dist_mask = dist < threshold
    normal = Normal(mu, sigma)
    # [BSZ, N, M, 10]
    loglik = normal.log_prob(dist.unsqueeze(-1))
    logprob = loglik + torch.log(pi)
    # [BSZ, N, M]
    prob = logprob.exp().sum(-1)
    # score = torch.stack([p[m].sum() for p, m in zip(prob, dist_mask)])
    score = (prob[dist_mask] / (dist[dist_mask] ** 2 + 1e-6)).sum()
    # score = prob[dist_mask].sum()
    return score


def prepare_log_data(mol_list, pocket_coords, distance_predict_tta, holo_distance_predict_tta, pi, mu, sigma):
    rst = []
    _pocket_coords = torch.from_numpy(pocket_coords)
    _distance_predict_tta = [torch.from_numpy(d) for d in distance_predict_tta]
    _holo_distance_predict_tta = [torch.from_numpy(d) for d in holo_distance_predict_tta]
    for mol in mol_list:
        coords = mol.GetConformer().GetPositions()
        _coords = torch.from_numpy(coords)
        loss = np.min([single_SF_loss(_coords, _pocket_coords, d1, d2).item() for d1, d2 in
                       zip(_distance_predict_tta, _holo_distance_predict_tta)])
        score = mdn_score(pi, mu, sigma, _coords, _pocket_coords).item()
        rst.append((coords, score, loss))
    rst = sorted(zip(rst, mol_list), key=lambda x: x[0][1] - 5 * x[0][2], reverse=True)
    log_data = [i[0] for i in rst]
    mol_list = [i[1] for i in rst]
    for mol, d in zip(mol_list, log_data):
        mol.SetProp('loss', f'{d[2]}')
        mol.SetProp('score', f'{d[1]}')
        mol.SetProp('agg_score', f'{d[2] - d[1]}')
    return log_data, mol_list


def save_sdf(mol_list, output_path, ):
    with Chem.SDWriter(output_path) as w:
        for i, mol in enumerate(mol_list):
            w.write(mol)


def result_log(dir_path, topk=1):
    ### result logging ###
    output_dir = os.path.join(dir_path, "cache")
    # output_dir = os.path.join(dir_path, "0302_v1")
    error_log_file = os.path.join(dir_path, "skipped_data_log.txt")  ## ls revise

    rmsd_results = []
    for path in glob.glob(os.path.join(output_dir, "*.docking.pkl")):
        (
            # bst_predict_coords,
            log_data,
            holo_coords,
            # bst_loss,
            smi,
            pocket,
            pocket_coords,
            mol
        ) = pd.read_pickle(path)
        # rmsd = min([rmsd_func(holo_coords, pred_coords[0]) for pred_coords in log_data][:topk])
        # rmsd = [rmsd_func(holo_coords, pred_coords[0]) for pred_coords in log_data]
        rmsd = get_symmetry_rmsd(mol, holo_coords, [i[0] for i in log_data])
        score = [40 * pred_coords[2] - 1 * pred_coords[1] for pred_coords in log_data]
        sort_list = [(r, s) for r, s in zip(rmsd, score)]
        sort_list = sorted(sort_list, key=lambda x: x[1])[:topk]
        ##---ls revise 
        # MODIFICATION START
        # 1. 检查 sort_list 是否为空
        if not sort_list:
            # 如果列表为空，执行以下操作
            
            # 2. 提取 PDB ID (或其他唯一标识符)
            # 假设文件名格式为 "pdb_id.docking.pkl"
            pdb_id = os.path.basename(path).split('.docking.pkl')[0]
            print(f"警告: {pdb_id} 的 sort_list 为空。跳过此数据并记录。")
            
            # 3. 将这个 pdb_id 输出到一个 txt 文件
            with open(error_log_file, 'a') as f: # 使用 'a' 模式来追加内容
                f.write(f"{pdb_id}\n")
            
            # 4. 使用 continue 跳过当前循环的剩余部分
            continue

        # 如果 sort_list 不为空，则正常执行下面的代码
        rmsd = min([i[0] for i in sort_list])
        rmsd_results.append(rmsd)
        
        # MODIFICATION END  ## ls revise end
        #rmsd = min([i[0] for i in sort_list])

        #rmsd_results.append(rmsd)
    rmsd_results = np.array(rmsd_results)
    print(f'==========TOP{topk}==========')
    print_results(rmsd_results)


def ana_result(dir_path):
    from scipy import stats
    output_dir = os.path.join(dir_path, "cache")
    # output_dir = os.path.join(dir_path, "0221-v1_25")
    results = []
    recalls = []
    fns = glob.glob(os.path.join(output_dir, "*.docking.pkl"))
    print(len(fns))
    for path in fns:
        (
            # bst_predict_coords,
            log_data,
            holo_coords,
            # bst_loss,
            smi,
            pocket,
            pocket_coords,
            mol
        ) = pd.read_pickle(path)
        # rmsd = np.array([rmsd_func(holo_coords, pred_coords[0]) for pred_coords in log_data])
        rmsd = get_symmetry_rmsd(mol, holo_coords, [i[0] for i in log_data])
        score = np.array([40 * pred_coords[2] - 1 * pred_coords[1] for pred_coords in log_data])
        # print(path)
        # print('\n'.join([f'{rmsd_func(holo_coords, pred_coords[0])}' for pred_coords in log_data]))
        results.append(stats.spearmanr(score, rmsd)[0])
        # threshold = sorted(rmsd)[2]
        # indices = np.argsort(score)
        # r_1 =

    print(f'mean:{np.mean(results)}, mid:{sorted(results)[len(results) // 2]}')


class MultiProcess():
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def dump(self, content):
        pocket = content[3]
        output_name = os.path.join(self.output_dir, "{}.pkl".format(pocket))
        try:
            os.remove(output_name)
        except:
            pass
        pd.to_pickle(content, output_name)
        return True

    def single_docking(self, pocket_name):
        input_name = os.path.join(self.output_dir, "{}.pkl".format(pocket_name))
        output_path = os.path.join(self.output_dir, "{}.docking.pkl".format(pocket_name))
        output_ligand_path = os.path.join(
            self.output_dir, "{}.ligand.sdf".format(pocket_name)
        )
        try:
            os.remove(output_path)
        except:
            pass
        try:
            os.remove(output_ligand_path)
        except:
            pass
        cmd = "python src/utils/coordinate_model.py --input {} --output {} --output-ligand {}".format(
            input_name, output_path, output_ligand_path
        )
        os.system(cmd)
        return True


def docking(raw_data_path, predict, nthreads):
    tta_times = 10
    (
        mol_dict,
        smi_list,
        pocket_list,
        pocket_coords_list,
        distance_predict_list,
        holo_distance_predict_list,
        holo_coords_list,
        holo_center_coords_list,
        pi_list,
        mu_list,
        sigma_list,
        init_coords_list,
    ) = docking_data_pre(raw_data_path, predict)
    iterations = ensemble_iterations(
        mol_dict,
        smi_list,
        pocket_list,
        pocket_coords_list,
        distance_predict_list,
        holo_distance_predict_list,
        holo_coords_list,
        holo_center_coords_list,
        pi_list,
        mu_list,
        sigma_list,
        init_coords_list,
        tta_times=tta_times,
    )

    sz = len(mol_dict) // tta_times
    new_pocket_list = pocket_list[::tta_times]
    output_dir = os.path.join(raw_data_path, "cache")
    os.makedirs(output_dir, exist_ok=True)
    MP = MultiProcess(output_dir)

    with Pool(4) as pool:
        for inner_output in tqdm(pool.imap(MP.dump, iterations), total=sz):
            if not inner_output:
                print("fail to dump")

    with Pool(nthreads) as pool:
        for inner_output in tqdm(
                pool.imap(MP.single_docking, new_pocket_list), total=len(new_pocket_list)
        ):
            if not inner_output:
                print("fail to docking")

    result_log(raw_data_path, topk=1)
    result_log(raw_data_path, topk=5)
    result_log(raw_data_path, topk=10)
    result_log(raw_data_path, topk=50)
    result_log(raw_data_path, topk=100)


class OptimizeConformer:
    def __init__(self, mol, true_mol, rotable_bonds, probe_id=-1, ref_id=-1, seed=None):
        super(OptimizeConformer, self).__init__()
        if seed:
            np.random.seed(seed)
        self.rotable_bonds = rotable_bonds
        self.mol = mol
        self.true_mol = true_mol
        self.probe_id = probe_id
        self.ref_id = ref_id

    def score_conformation(self, values):
        for i, r in enumerate(self.rotable_bonds):
            SetDihedral(self.mol.GetConformer(self.probe_id), r, values[i])
        return AllChem.AlignMol(self.mol, self.true_mol, self.probe_id, self.ref_id)


class GenerateConformer:
    def __init__(self, mol, rotable_bonds, pocket_coords, pred_cross_dist, pred_holo_dist, true_mol=None, seed=None):
        if seed:
            np.random.seed(seed)
        self.mol = mol
        if true_mol is None:
            self.mol_template = copy.deepcopy(mol)
        else:
            self.mol_template = true_mol
        self.rotable_bonds = rotable_bonds
        self.pocket_coords = pocket_coords
        self.pred_cross_dist = torch.from_numpy(pred_cross_dist)
        self.pred_holo_dist = torch.from_numpy(pred_holo_dist)
        self.min_loss = float('inf')

    def score_conformation(self, values):
        for i, r in enumerate(self.rotable_bonds):
            SetDihedral(self.mol.GetConformer(), r, values[i])
        rmsd = AllChem.AlignMol(self.mol, self.mol_template)
        # 旋转 + 平移
        # 计算loss并return
        angles = torch.from_numpy(values[-6:-3])
        bias = torch.from_numpy(values[-3:])
        coords = torch.from_numpy(self.mol.GetConformer().GetPositions())
        matrix = axis_angle_to_matrix(angles)
        new_coords = coords @ matrix + bias
        # set_coord(self.mol, new_coords.cpu().data.numpy())
        loss = single_SF_loss(new_coords, self.pocket_coords, self.pred_cross_dist, self.pred_holo_dist).item()
        if loss < self.min_loss:
            self.min_loss = loss
        return loss


def apply_changes(mol, values, rotable_bonds, conf_id):
    opt_mol = copy.copy(mol)
    [SetDihedral(opt_mol.GetConformer(conf_id), rotable_bonds[r], values[r]) for r in range(len(rotable_bonds))]
    return opt_mol


def optimize_rotatable_bonds(mol, true_mol, rotable_bonds, probe_id=-1, ref_id=-1, seed=0, popsize=15, maxiter=500,
                             mutation=(0.5, 1), recombination=0.8, use_de=True):
    if use_de:
        opt = OptimizeConformer(mol, true_mol, rotable_bonds, seed=seed, probe_id=probe_id, ref_id=ref_id)
        max_bound = [np.pi] * len(opt.rotable_bonds)
        min_bound = [-np.pi] * len(opt.rotable_bonds)
        bounds = (min_bound, max_bound)
        bounds = list(zip(bounds[0], bounds[1]))

        # Optimize conformations
        result = differential_evolution(opt.score_conformation, bounds,
                                        maxiter=maxiter, popsize=popsize,
                                        mutation=mutation, recombination=recombination, disp=False, seed=seed)
        opt_mol = apply_changes(opt.mol, result['x'], opt.rotable_bonds, conf_id=probe_id)
        return opt_mol
    else:
        values = [GetDihedral(true_mol.GetConformer(), rotable_bond) for rotable_bond in rotable_bonds]
        for i, r in enumerate(rotable_bonds):
            SetDihedral(mol.GetConformer(), r, values[i])
        AllChem.AlignMol(mol, true_mol, probe_id, ref_id)
        return mol


def optimize_conformer(mol, rotable_bonds, pocket_coords, pred_cross_dist, pred_holo_dist, true_mol=None, seed=42,
                       popsize=15,
                       maxiter=50, mutation=(0.5, 1), recombination=0.8):
    opt = GenerateConformer(mol, rotable_bonds, pocket_coords, pred_cross_dist, pred_holo_dist, true_mol=true_mol,
                            seed=seed)
    max_bound = [np.pi] * (len(opt.rotable_bonds) + 3)
    min_bound = [-np.pi] * (len(opt.rotable_bonds) + 3)
    bounds = (min_bound, max_bound)
    bounds = list(zip(bounds[0], bounds[1])) + [(-5.0, 5.0)] * 3

    # Optimize conformations
    result = differential_evolution(opt.score_conformation, bounds,
                                    maxiter=maxiter, popsize=popsize,
                                    mutation=mutation, recombination=recombination, disp=False, seed=seed)
    values = result['x']
    for i, r in enumerate(rotable_bonds):
        SetDihedral(opt.mol.GetConformer(), r, values[i])
    rmsd = AllChem.AlignMol(opt.mol, opt.mol_template)
    angles = torch.from_numpy(values[-6:-3])
    bias = torch.from_numpy(values[-3:])
    coords = torch.from_numpy(opt.mol.GetConformer().GetPositions())
    matrix = axis_angle_to_matrix(angles)
    new_coords = coords @ matrix + bias
    opt_mol = set_coord(opt.mol, new_coords.cpu().data.numpy())
    return opt_mol


def align_conformer(mol_rdkit, mol):
    mol.AddConformer(mol_rdkit.GetConformer())
    rms_list = []
    AllChem.AlignMolConformers(mol, RMSlist=rms_list)
    mol_rdkit.RemoveAllConformers()
    mol_rdkit.AddConformer(mol.GetConformers()[1])
    return mol_rdkit


def get_symmetry_rmsd(mol, coords1, coords2, mol2=None):
    mol = molecule.Molecule.from_rdkit(mol)
    mol2 = molecule.Molecule.from_rdkit(mol2) if mol2 is not None else mol2
    mol2_atomicnums = mol2.atomicnums if mol2 is not None else mol.atomicnums
    mol2_adjacency_matrix = mol2.adjacency_matrix if mol2 is not None else mol.adjacency_matrix
    RMSD = rmsd.symmrmsd(
        coords1,
        coords2,
        mol.atomicnums,
        mol2_atomicnums,
        mol.adjacency_matrix,
        mol2_adjacency_matrix,
    )
    return RMSD


def dist2coord(raw_data_path, nthreads=24):
    pocket_list = glob.glob(os.path.join(raw_data_path, 'cache', '*.pkl'))
    new_pocket_list = list({p.split('.')[0] for p in pocket_list})
    output_dir = os.path.join(raw_data_path, "cache")
    os.makedirs(output_dir, exist_ok=True)
    MP = MultiProcess(output_dir)

    with Pool(nthreads) as pool:
        for inner_output in tqdm(
                pool.imap(MP.single_docking, new_pocket_list), total=len(new_pocket_list)
        ):
            if not inner_output:
                print("fail to docking")

    result_log(raw_data_path, topk=1)
    result_log(raw_data_path, topk=5)
    result_log(raw_data_path, topk=10)
    result_log(raw_data_path, topk=25)
    result_log(raw_data_path, topk=50)


def get_k_nearest_mask(target: torch.Tensor, k=10):
    with torch.no_grad():
        mask = target.eq(0)
        _target = target.clone()
        _target[mask] = _target[mask] + 1000
        indices = _target.topk(k, dim=-1, largest=False)[1]
        rst_mask = torch.zeros_like(target, device=target.device, dtype=torch.bool)
        ind_list = itertools.product(*[list(range(i)) for i in target.shape[:-1]])
        for item in ind_list:
            rst_mask[item][indices[item]] = True
        rst_mask = rst_mask & (~mask)
    return rst_mask
## ls revise
def extract_carsidock_pocket(pdb_file, ligand_file, keep_water=True, distance=6):
    with open(pdb_file, 'r') as pdb_file:
        # supp = Chem.SDMolSupplier(ligand_file)
        # ligand = Chem.RemoveAllHs(supp[0])
        ligand = read_mol(ligand_file)
        if ligand is None:
            ligand = read_mol(ligand_file, sanitize=False)
            positions = ligand.GetConformer().GetPositions()
            atoms = np.array([a.GetSymbol() for a in ligand.GetAtoms()])
            positions = positions[atoms!='H']
        else:
            if ligand_file.endswith('.sdf'):
                ligand = ligand[0]
            ligand = Chem.RemoveHs(ligand, sanitize=True, implicitOnly=True)
            positions = ligand.GetConformer().GetPositions()
        ##----ls revise for RNA-----   
        #if keep_water:
        #    protein = prody.parsePDBStream(pdb_file).select('protein or water')
        #else:
        #    protein = prody.parsePDBStream(pdb_file).select('protein')
        #selected = protein.select(f'same residue as within {distance} of ligand',
        #                          ligand=positions)
        receptor = prody.parsePDBStream(pdb_file)
        selected = receptor.select(f'same residue as within {distance} of ligand',
                                  ligand=positions)
        ##----ls revise for RNA-----   
        f = io.StringIO()
        prody.writePDBStream(f, selected)
        pocket = Chem.MolFromPDBBlock(f.getvalue(), sanitize=False, removeHs=True)
        # pocket = Chem.RemoveHs(pocket)
        # pocket_pdb_out='outputs/pocket.pdb'
        # Chem.MolToPDBFile(pocket, pocket_pdb_out)
    return pocket, ligand

def extract_pocket_pro(pdb_file, positions: np, distance = 6, sanitize=False, del_water=False, del_ion=True):
    pdb_str = Path(pdb_file).read_text()
    protein = prody.parsePDBStream(io.StringIO(pdb_str)).select('protein or water')
    selected = protein.select(
        f'same residue as within {distance} of ligand',
        ligand=positions,
    )
    f = io.StringIO()
    prody.writePDBStream(f, selected)
    pocket_str = f.getvalue()
    pdb_content = []
    for line in io.StringIO(pocket_str).readlines():
        if del_water and line.startswith('HETATM') and line[17:20] =='HOH':
            pass
        elif del_ion and len(line[17:20].strip()) == 2:
            pass
        else:
            pdb_content.append(line)

    if sanitize:
        prot = Chem.MolFromPDBBlock(''.join(pdb_content), sanitize=False)
        problems = Chem.DetectChemistryProblems(prot)
        if problems:
            raise ValueError(f'pocket: {[problem.Message() for problem in problems]}')
        else:
            pocket = Chem.MolFromPDBBlock(''.join(pdb_content))
            pocket = Chem.RemoveHs(pocket)
            return pocket
    else:
        pocket = Chem.MolFromPDBBlock(''.join(pdb_content), sanitize=False, removeHs=True)
        pocket = Chem.RemoveHs(pocket, sanitize=False)
        return pocket
        
def extract_pocket(pdb_file, positions: np.ndarray,
                   distance: float = 6.0,
                   sanitize: bool = False,
                   del_water: bool = False,
                   del_ion: bool = True):
    pdb_str = Path(pdb_file).read_text()

    # 1. 先解析整个结构
    ag = prody.parsePDBStream(io.StringIO(pdb_str))

    # ★ 对 RNA/核酸来说，不能只选 protein，要把 nucleic 也选上
    #    如果你以后还会混合蛋白 + RNA，可以用 'protein or nucleic or water'
    receptor = ag.select('protein or nucleic or water')
    if receptor is None or receptor.numAtoms() == 0:
        raise ValueError(
            f"[extract_pocket] 在 {pdb_file} 中没有选到 protein/nucleic/water 原子，"
            f"请检查 PDB 内容或选择字符串。"
        )

    # 2. 以 ligand 坐标为中心，选一定距离内的残基
    selected = receptor.select(
        f"same residue as within {distance} of ligand",
        ligand=positions,
    )
    if selected is None or selected.numAtoms() == 0:
        raise ValueError(
            f"[extract_pocket] 在 {pdb_file} 中，距离配体 <= {distance} Å 没有选到任何 pocket 原子，"
            f"可能是 PDB 和参考配体不在同一构象/坐标系，或者 distance 太小。"
        )

    # 3. 写成 PDB 格式字符串
    f = io.StringIO()
    prody.writePDBStream(f, selected)
    pocket_str = f.getvalue()

    # 4. 过滤水和离子（如果需要）
    pdb_content = []
    for line in io.StringIO(pocket_str).readlines():
        if del_water and line.startswith('HETATM') and line[17:20] == 'HOH':
            pass
        # 很多金属/离子是两个字母的 residue name，这里按你原来的规则过滤
        elif del_ion and len(line[17:20].strip()) == 2:
            pass
        else:
            pdb_content.append(line)

    if not pdb_content:
        raise ValueError(
            "[extract_pocket] 过滤 water/ion 后 pocket 为空，"
            "请尝试关闭 del_water/del_ion 或调大距离。"
        )

    pdb_block = ''.join(pdb_content)

    # 5. 交给 RDKit 解析
    if sanitize:
        prot = Chem.MolFromPDBBlock(pdb_block, sanitize=False)
        if prot is None:
            raise ValueError("[extract_pocket] RDKit 无法解析 pocket 的 PDB block（sanitize=True 分支，prot=None）。")

        problems = Chem.DetectChemistryProblems(prot)
        if problems:
            msgs = [p.Message() for p in problems]
            raise ValueError(f"[extract_pocket] pocket 化学问题: {msgs}")
        pocket = Chem.MolFromPDBBlock(pdb_block)  # 默认 sanitize=True
        if pocket is None:
            raise ValueError("[extract_pocket] RDKit 无法在 sanitize=True 下构建 pocket 分子。")
        pocket = Chem.RemoveHs(pocket)
    else:
        pocket = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=True)
        if pocket is None:
            raise ValueError("[extract_pocket] RDKit 无法在 sanitize=False 下构建 pocket 分子。")
        pocket = Chem.RemoveHs(pocket, sanitize=False)

    return pocket

        
def read_mol(mol_path, sanitize=True):
    if re.search(r'.pdb$', mol_path):
        mol = Chem.MolFromPDBFile(mol_path, sanitize=sanitize, removeHs=True)
    elif re.search(r'.mol2$', mol_path):
        mol = Chem.MolFromMol2File(mol_path, sanitize=sanitize, removeHs=True)
    elif re.search(r'.mol$', mol_path):
        mol = Chem.MolFromMolFile(mol_path, sanitize=sanitize, removeHs=True)
    else:
        mol = Chem.SDMolSupplier(mol_path, sanitize=sanitize, removeHs=True)
    return mol

def read_pocket(pdb_path):
    """
    read pocket from .pdb file.
    """
    p_mol = read_mol(pdb_path)
    return p_mol

if __name__ == '__main__':
    dist2coord('/mnt/d/workspace/data')

