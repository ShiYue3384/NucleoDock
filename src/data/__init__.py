import os
from src.utils.dictionary import Dictionary
from torch.utils.data import DataLoader
from . import algos


def get_datasets(cfg, splits, load_dataset, dictionary=None):
    if dictionary is None:
        dictionary = Dictionary.load(os.path.join(cfg.data.path, cfg.data.dict_name))
        mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        # 加10个无用token以备不时之需
        # for i in range(10):
        #     dictionary.add_symbol(f"[UNUSED{i}]", is_special=True)
    datasets = dict()
    for split in splits:
        datasets[split] = load_dataset(cfg, split, dictionary, mask_idx=dictionary.index('[MASK]'))
    return datasets, dictionary


def get_dataloader(cfg, datasets):
    dataloaders = dict()
    for key in datasets.keys():
        is_train = 'train' in key
        dataloaders[key] = DataLoader(dataset=datasets[key],
                                      batch_size=cfg.SOLVER.TRAIN_BSZ if is_train else cfg.SOLVER.VALID_BSZ,
                                      shuffle=is_train,
                                      drop_last=False,
                                      collate_fn=datasets[key].collater,
                                      num_workers=cfg.DATALOADER.NUM_WORKERS,
                                      pin_memory=False)
    return dataloaders