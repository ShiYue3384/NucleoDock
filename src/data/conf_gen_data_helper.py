# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import contextlib
from typing import Optional

import numpy as np
from .unicore import (
    Dictionary,
    NestedDictionaryDataset,
    LMDBDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    SortDataset,
    TokenizeDataset,
    RightPadDataset2D,
    RawArrayDataset,
    FromNumpyDataset,
)
from .unimol_datasets import (
    KeyDataset,
    DistanceDataset,
    EdgeTypeDataset,
    NormalizeDataset,
    RightPadDatasetCoord,
    ConformerSampleConfGDataset,
    ConformerSampleConfGV2Dataset,
    data_utils,
)

logger = logging.getLogger(__name__)


def prepend_and_append(dataset, pre_token, app_token):
    dataset = PrependTokenDataset(dataset, pre_token)
    return AppendTokenDataset(dataset, app_token)

def load_dataset(cfg, split, dictionary, **kwargs):
    """Load a given dataset split.
    Args:
        cfg: global config
        split (str): name of the data scoure (e.g., bppp)
        dictionary: dictionary
    """
    split_path = os.path.join(cfg.data.path, split + ".lmdb")
    dataset = LMDBDataset(split_path)
    smi_dataset = KeyDataset(dataset, "smi")
    src_dataset = KeyDataset(dataset, "atoms")
    if not split.startswith("test"):
        sample_dataset = ConformerSampleConfGV2Dataset(
            dataset,
            cfg.MODEL.SEED,
            "atoms",
            "coordinates",
            "target",
            cfg.data.beta,
            cfg.data.smooth,
            cfg.data.topN,
        )
    else:
        sample_dataset = ConformerSampleConfGDataset(
            dataset, cfg.MODEL.SEED, "atoms", "coordinates", "target"
        )
    sample_dataset = NormalizeDataset(sample_dataset, "coordinates")
    sample_dataset = NormalizeDataset(sample_dataset, "target")
    src_dataset = TokenizeDataset(
        src_dataset, dictionary, max_seq_len=512
    )
    coord_dataset = KeyDataset(sample_dataset, "coordinates")
    tgt_coord_dataset = KeyDataset(sample_dataset, "target")



    tgt_coord_dataset = FromNumpyDataset(tgt_coord_dataset)
    tgt_coord_dataset = prepend_and_append(tgt_coord_dataset, 0.0, 0.0)
    tgt_distance_dataset = DistanceDataset(tgt_coord_dataset)

    src_dataset = prepend_and_append(
        src_dataset, dictionary.bos(), dictionary.eos()
    )
    edge_type = EdgeTypeDataset(src_dataset, len(dictionary))
    coord_dataset = FromNumpyDataset(coord_dataset)
    coord_dataset = prepend_and_append(coord_dataset, 0.0, 0.0)
    distance_dataset = DistanceDataset(coord_dataset)

    nest_dataset = NestedDictionaryDataset(
        {
            "net_input": {
                "src_tokens": RightPadDataset(
                    src_dataset,
                    pad_idx=dictionary.pad(),
                ),
                "src_coord": RightPadDatasetCoord(
                    coord_dataset,
                    pad_idx=0,
                ),
                "src_distance": RightPadDataset2D(
                    distance_dataset,
                    pad_idx=0,
                ),
                "src_edge_type": RightPadDataset2D(
                    edge_type,
                    pad_idx=0,
                ),
            },
            "target": {
                "coord_target": RightPadDatasetCoord(
                    tgt_coord_dataset,
                    pad_idx=0,
                ),
                "distance_target": RightPadDataset2D(
                    tgt_distance_dataset,
                    pad_idx=0,
                ),
            },
            "smi_name": RawArrayDataset(smi_dataset),
        },
)
    return nest_dataset
