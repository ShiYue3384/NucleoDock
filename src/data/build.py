from .get_paths import *
from .loaders import big_text_image_pair_loader, get_text_loader
from .collators import DataCollatorForCLIP, DataCollatorForOnlyMask
from transformers import AutoTokenizer
from transformers.models.clip.feature_extraction_clip import CLIPFeatureExtractor
from .tokenize_funcs import IdsDataTokenizeFunc


def make_loaders(cfg):
    if cfg.MODEL.TASK.lower() == 'clip':
        fps = get_image_json_fps(cfg.DATASETS.TRAIN, cfg.DATALOADER.NUM_FILES)
        tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.TOKENIZER_NAME)
        data_collator = DataCollatorForCLIP(tokenizer)
        feature_extractor = CLIPFeatureExtractor.from_pretrained(cfg.MODEL.FEATURE_EXTRACTOR_NAME)
        loader = big_text_image_pair_loader(cfg, tokenizer, feature_extractor, data_collator, fps=fps)
        return loader
    elif cfg.MODEL.TASK.lower() == 'text_data2vec':
        fps = get_wiki_ids_fps(cfg.DATASETS.TRAIN, cfg.DATALOADER.NUM_FILES)
        tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.TOKENIZER_NAME)
        data_collator = DataCollatorForOnlyMask(tokenizer, cfg.MODEL.MLM_PROB)
        tokenize_func = IdsDataTokenizeFunc()
        loader = get_text_loader(cfg, fps, data_collator, tokenize_func)
        return loader
