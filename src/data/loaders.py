from .datasets import BigDataTokenPairDataset
from torch.utils.data import DataLoader
from datasets import load_dataset


def big_text_image_pair_loader(cfg, tokenizer, feature_extractor, data_collator, fps, **kwargs):
    train_dataset = BigDataTokenPairDataset(cfg, fps, '/data1122/img', tokenizer, feature_extractor)
    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  collate_fn=data_collator,
                                  batch_size=cfg.SOLVER.TRAIN_BSZ,
                                  num_workers=cfg.DATALOADER.NUM_WORKERS,
                                  **kwargs)
    return train_dataloader


def get_text_loader(cfg, fps, data_collator, tokenize_func, **kwargs):
    tokenize_func = tokenize_func
    raw_datasets = load_dataset('text', data_files=fps)
    train_dataset = raw_datasets.map(tokenize_func, batched=True, num_proc=cfg.DATALOADER.NUM_PROC,
                                     remove_columns=['text'])['train']
    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  collate_fn=data_collator,
                                  batch_size=cfg.SOLVER.TRAIN_BSZ,
                                  num_workers=cfg.DATALOADER.NUM_WORKERS,
                                  **kwargs)
    return train_dataloader
