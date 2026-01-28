# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

from .unicore import (
    Dictionary,
    NestedDictionaryDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    TokenizeDataset,
    RightPadDataset2D,
    RawArrayDataset,
    FromNumpyDataset,
    EpochShuffleDataset,
)
from .unicore import (
    LMDBDataset,
    KeyDataset,
    ConformerSampleDockingPoseDataset,
    DistanceDataset,
    EdgeTypeDataset,
    NormalizeDataset,
    RightPadDatasetCoord,
    CrossDistanceDataset,
    NormalizeDockingPoseDataset,
    TTADockingPoseDataset,
    RightPadDatasetCross2D,
    CroppingPocketDockingPoseDataset,
    PrependAndAppend2DDataset,
    RemoveHydrogenPocketDataset,
)
from .unimol_datasets import AffinityDataset, HopMatrixDataset
from ..utils.utils import get_abs_path

logger = logging.getLogger(__name__)


def load_dataset(cfg, split, mol_dictionary, pocket_dictionary, **kwargs):
    """Load a given dataset split.
    'smi','pocket','atoms','coordinates','pocket_atoms','pocket_coordinates','holo_coordinates','holo_pocket_coordinates','scaffold'
    """
    data_path = os.path.join(cfg.data.path, split + ".lmdb")
    dataset = LMDBDataset(data_path)
    mol_dataset = KeyDataset(dataset, 'mol_list')
    if split.startswith("train"):
        smi_dataset = KeyDataset(dataset, "smi")
        poc_dataset = KeyDataset(dataset, "pocket")
        dataset = ConformerSampleDockingPoseDataset(
            dataset,
            cfg.MODEL.SEED,
            "atoms",
            "coordinates",
            "pocket_atoms",
            "pocket_coordinates",
            "holo_coordinates",
            "holo_pocket_coordinates",
            # "new_pocket_atoms",
            # "new_pocket_coordinates",
            # "holo_coordinates",
            # "new_pocket_coordinates",
            True,
        )
        hop_mat_dataset = HopMatrixDataset(mol_dataset, 'train')
    else:
        dataset = TTADockingPoseDataset(
            dataset,
            "atoms",
            "coordinates",
            "pocket_atoms",
            "pocket_coordinates",
            "holo_coordinates",
            "holo_pocket_coordinates",
            # "new_pocket_atoms",
            # "new_pocket_coordinates",
            # "holo_coordinates",
            # "new_pocket_coordinates",
            True,
            cfg.data.conf_size,
        )
        smi_dataset = KeyDataset(dataset, "smi")
        poc_dataset = KeyDataset(dataset, "pocket")
        hop_mat_dataset = HopMatrixDataset(mol_dataset, split=split, conf_size=cfg.data.conf_size)

    def PrependAndAppend(dataset, pre_token, app_token):
        dataset = PrependTokenDataset(dataset, pre_token)
        return AppendTokenDataset(dataset, app_token)

    hop_mat_dataset = PrependAndAppend2DDataset(hop_mat_dataset, 127)

    dataset = RemoveHydrogenPocketDataset(
        dataset,
        "pocket_atoms",
        "pocket_coordinates",
        "holo_pocket_coordinates",
        True,
        True,
    )
    dataset = CroppingPocketDockingPoseDataset(
        dataset,
        cfg.MODEL.SEED,
        "pocket_atoms",
        "pocket_coordinates",
        "holo_pocket_coordinates",
        cfg.data.max_pocket_atoms,
    )
    dataset = RemoveHydrogenPocketDataset(
        dataset, "atoms", "coordinates", "holo_coordinates", True, True
    )

    apo_dataset = NormalizeDataset(dataset, "coordinates")
    apo_dataset = NormalizeDataset(apo_dataset, "pocket_coordinates")

    src_dataset = KeyDataset(apo_dataset, "atoms")
    src_dataset = TokenizeDataset(
        src_dataset, mol_dictionary, max_seq_len=cfg.data.max_seq_len
    )
    coord_dataset = KeyDataset(apo_dataset, "coordinates")
    src_dataset = PrependAndAppend(
        src_dataset, mol_dictionary.bos(), mol_dictionary.eos()
    )
    edge_type = EdgeTypeDataset(src_dataset, len(mol_dictionary))
    coord_dataset = FromNumpyDataset(coord_dataset)
    distance_dataset = DistanceDataset(coord_dataset)
    coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
    distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)

    src_pocket_dataset = KeyDataset(apo_dataset, "pocket_atoms")
    src_pocket_dataset = TokenizeDataset(
        src_pocket_dataset,
        pocket_dictionary,
        max_seq_len=cfg.data.max_seq_len,
    )
    coord_pocket_dataset = KeyDataset(apo_dataset, "pocket_coordinates")
    src_pocket_dataset = PrependAndAppend(
        src_pocket_dataset,
        pocket_dictionary.bos(),
        pocket_dictionary.eos(),
    )
    pocket_edge_type = EdgeTypeDataset(
        src_pocket_dataset, len(pocket_dictionary)
    )
    coord_pocket_dataset = FromNumpyDataset(coord_pocket_dataset)
    distance_pocket_dataset = DistanceDataset(coord_pocket_dataset)
    coord_pocket_dataset = PrependAndAppend(coord_pocket_dataset, 0.0, 0.0)
    distance_pocket_dataset = PrependAndAppend2DDataset(
        distance_pocket_dataset, 0.0
    )

    holo_dataset = NormalizeDockingPoseDataset(
        dataset,
        "holo_coordinates",
        "holo_pocket_coordinates",
        "holo_center_coordinates",
    )
    holo_coord_dataset = KeyDataset(holo_dataset, "holo_coordinates")
    holo_coord_dataset = FromNumpyDataset(holo_coord_dataset)
    holo_coord_pocket_dataset = KeyDataset(holo_dataset, "holo_pocket_coordinates")
    holo_coord_pocket_dataset = FromNumpyDataset(holo_coord_pocket_dataset)

    holo_cross_distance_dataset = CrossDistanceDataset(
        holo_coord_dataset, holo_coord_pocket_dataset
    )

    holo_distance_dataset = DistanceDataset(holo_coord_dataset)
    holo_coord_dataset = PrependAndAppend(holo_coord_dataset, 0.0, 0.0)
    holo_distance_dataset = PrependAndAppend2DDataset(holo_distance_dataset, 0.0)
    holo_coord_pocket_dataset = PrependAndAppend(
        holo_coord_pocket_dataset, 0.0, 0.0
    )
    holo_cross_distance_dataset = PrependAndAppend2DDataset(
        holo_cross_distance_dataset, 0.0
    )

    holo_center_coordinates = KeyDataset(holo_dataset, "holo_center_coordinates")
    holo_center_coordinates = FromNumpyDataset(holo_center_coordinates)

    nest_dataset = NestedDictionaryDataset(
        {
            "net_input": {
                "mol_src_tokens": RightPadDataset(
                    src_dataset,
                    pad_idx=mol_dictionary.pad(),
                ),
                "mol_src_distance": RightPadDataset2D(
                    distance_dataset,
                    pad_idx=0,
                ),
                "mol_src_edge_type": RightPadDataset2D(
                    edge_type,
                    pad_idx=0,
                ),
                "pocket_src_tokens": RightPadDataset(
                    src_pocket_dataset,
                    pad_idx=pocket_dictionary.pad(),
                ),
                "pocket_src_distance": RightPadDataset2D(
                    distance_pocket_dataset,
                    pad_idx=0,
                ),
                "pocket_src_edge_type": RightPadDataset2D(
                    pocket_edge_type,
                    pad_idx=0,
                ),
                "pocket_src_coord": RightPadDatasetCoord(
                    coord_pocket_dataset,
                    pad_idx=0,
                ),
                'holo_init_coord': RightPadDatasetCoord(coord_dataset, pad_idx=0)
            },
            "target": {
                "distance_target": RightPadDatasetCross2D(
                    holo_cross_distance_dataset, pad_idx=0
                ),
                "holo_coord": RightPadDatasetCoord(holo_coord_dataset, pad_idx=0),
                "holo_distance_target": RightPadDataset2D(
                    holo_distance_dataset, pad_idx=0
                ),
                'score': RawArrayDataset(AffinityDataset(poc_dataset, get_abs_path(cfg.data.affinity))),
                'hop_matrix': RightPadDataset2D(hop_mat_dataset, pad_idx=127)
            },
            "smi_name": RawArrayDataset(smi_dataset),
            "pocket_name": RawArrayDataset(poc_dataset),
            "holo_center_coordinates": RightPadDataset(
                holo_center_coordinates,
                pad_idx=0,
            ),
        },
    )
    return nest_dataset
