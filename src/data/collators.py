from io import StringIO
import random
from tkinter import N
import numpy as np
import torch
import copy
from io import StringIO

from transformers import PreTrainedTokenizerBase


class DataCollatorForCLIP(object):

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, examples):
        examples = [e for e in examples if e is not None]
        pixel_values = [e.pop('pixel_values') for e in examples]
        inputs = self.tokenizer.pad(examples, return_tensors='pt')
        pixel_values = torch.tensor(np.array(pixel_values))
        inputs['pixel_values'] = pixel_values.squeeze(1)
        return inputs


class DataCollatorForComC(object):

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, examples):
        examples = [e for e in examples if e is not None]
        pixel_values = [e.pop('pixel_values') for e in examples]
        bsz = len(examples)
        seq_len = max([2 * len(e['input_ids']) - 1 for e in examples])
        attention_mask = torch.zeros(bsz, seq_len, seq_len, dtype=torch.long)
        labels = []

        for i, example in enumerate(examples):
            example.pop('token_type_ids')
            text_len = len(example.pop('attention_mask'))
            example['input_ids'] = example['input_ids'] + example['input_ids'][1:]
            expanded_text_len = len(example['input_ids'])
            am = torch.tril(torch.ones(expanded_text_len, expanded_text_len, dtype=torch.long))
            am[:text_len, :text_len] = 1
            am[text_len:, 1:text_len] = 0
            attention_mask[i][:expanded_text_len, :expanded_text_len] = am
            label = [-100] * text_len + example['input_ids'][1:text_len]
            labels.append(label)
            ###### eg. text_len = 3, expanded_text_len = 5
            # am:
            # [[1, 1, 1, 0, 0],
            #  [1, 1, 1, 0, 0],
            #  [1, 1, 1, 0, 0],
            #  [1, 0, 0, 1, 0],
            #  [1, 0, 0, 1, 1]]

        inputs = self.tokenizer.pad(examples, return_tensors='pt')
        pixel_values = torch.tensor(np.array(pixel_values))
        inputs['pixel_values'] = pixel_values.squeeze(1)
        inputs['attention_mask'] = attention_mask
        inputs['labels'] = torch.tensor(np.array(labels), dtype=torch.long)
        return inputs


class DataCollatorForUniLMv2(object):

    def __init__(self, tokenizer: PreTrainedTokenizerBase, mlm_prob=0.15):
        self.tokenizer = tokenizer
        self.mlm_prob = mlm_prob

    def __call__(self, examples):
        input_ids = []
        attend_mask = []
        position_ids = []
        # TODO: add token_type_ids
        # token_type_ids = []
        labels = []

        inputs = [e['input_ids'] for e in examples]
        for i in inputs:
            mask_id = self.get_span_mask_ids(len(i))
            outputs = self.get_output(i, mask_id)
            input_ids.append(outputs[0])
            labels.append(outputs[1])
            position_ids.append(outputs[2])
            attend_mask.append(outputs[3])

        batch = {'input_ids': input_ids}
        batch = self.tokenizer.pad(batch, return_tensors='pt', pad_to_multiple_of=8, padding=True, return_attention_mask=False)
        length = batch['input_ids'].shape[1]
        attention_mask = torch.zeros(batch['input_ids'].shape[0], length, length, dtype=torch.int)
        for i, mask in enumerate(attend_mask):
            attention_mask[i][:mask.shape[0], :mask.shape[1]] += mask
        position_ids = [pids + [0] * (length - len(pids)) for pids in position_ids]
        position_ids = torch.tensor(position_ids, dtype=torch.int)
        labels = [label + [-100] * (length - len(label)) for label in labels]
        labels = torch.tensor(labels, dtype=torch.long)
        batch['attention_mask'] = attention_mask
        batch['position_ids'] = position_ids
        batch['labels'] = labels
        batch['token_type_ids'] = torch.zeros_like(batch['input_ids'], dtype=torch.int)
        return batch

    def get_output(self, input_ids, mask_ids):
        """
        由于使用了绝对位置编码，故统一将[MASK]及[P]滞后，排列顺序为tokens，masks，pseudo_masks, 该实现在使用绝对位置编码时与原文的实现等价。
        :param input_ids:
        :param mask_ids:
        :return:
        """
        length = len(input_ids) + len(mask_ids) * 2
        mask_matrix = torch.eye(length, dtype=torch.int)
        mask_set = set(mask_ids)
        no_mask_ids = [i for i in range(len(input_ids)) if i not in mask_set]
        # 所有未被mask的token均会被attend
        mask_matrix[:, no_mask_ids] = 1
        # 所有[MASK]标签均会被attend
        # mask_matrix[:, len(input_ids):len(input_ids) + len(mask_ids)] = 1
        # 修改：受MAE启发，所有[MASK]标签均不被Attend
        mask_matrix[:, len(input_ids):len(input_ids) + len(mask_ids)] = 0
        mask_matrix = self.pseudo_mask(mask_matrix, mask_ids, len(input_ids))

        position_ids = list(range(len(input_ids))) + mask_ids + mask_ids
        labels = [iid for i, iid in enumerate(input_ids) if i in mask_set]
        labels = [-100] * len(input_ids) + labels + labels
        # labels = [-100] * len(input_ids) + labels + [-100] * len(labels)
        mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        pmask_id = self.tokenizer.convert_tokens_to_ids('[unused1]')
        input_ids = input_ids + self.replace_inputs(input_ids, mask_ids, mask_id) + self.replace_inputs(
            input_ids, mask_ids, pmask_id)
        return input_ids, labels, position_ids, mask_matrix

    def get_span_mask_ids(self, length):
        mask_ids = set()
        num_mask = length * self.mlm_prob
        while len(mask_ids) < num_mask:
            p = random.randint(1, length - 1)
            # 原文的span mask会有较大概率mask较长，造成模型困惑度过高，此处有些许调整
            l = random.randint(2, 6) if random.random() < 0.4 else 1
            # l = random.choices([1, 2, 3, 4], [0.4, 0.3, 0.2, 0.1])[0]
            if p + l - 1 > length:
                mask_ids.update(set(range(p, length)))
            else:
                mask_ids.update(set(range(p, p + l - 1)))
        return sorted(list(mask_ids))

    def pseudo_mask(self, mask_matrix, mask_ids, input_length):
        segments = []
        for mid in mask_ids:
            if len(segments) == 0:
                segments.append([mid])
            elif segments[-1][-1] + 1 != mid:
                segments.append([mid])
            else:
                segments[-1].append(mid)
        random.shuffle(segments)
        attended_ids = []
        for segment in segments:
            start_idx = input_length + len(mask_ids) + mask_ids.index(segment[0])
            end_idx = start_idx + len(segment)
            if len(attended_ids) > 0:
                # 当前预测的span应attend到历史已预测的token，以备下一轮
                mask_matrix[segment[0]:segment[-1] + 1, attended_ids] = 1
            # 当前预测的token应相互之间attend，以备下一轮
            mask_matrix[segment[0]:segment[-1] + 1, segment[0]:segment[-1] + 1] = 1
            # [P]标签应attend到历史已预测的token
            mask_matrix[start_idx:end_idx, attended_ids] = 1
            # 当前span之间的[P]标签互相attend
            mask_matrix[start_idx:end_idx, start_idx:end_idx] = 1
            attended_ids += segment
        return mask_matrix

    def replace_inputs(self, input_ids, mask_ids, mask_token):
        probs = [random.random() for _ in mask_ids]

        def _select_token(input_id, prob):
            if prob < 0.8:
                return mask_token
            if prob < 0.9:
                return input_id
            return random.randint(100, len(self.tokenizer) - 1)

        return [_select_token(iid, prob) for iid, prob in zip(input_ids, probs)]


class DataCollatorForOnlyMask(object):

    def __init__(self, tokenizer: PreTrainedTokenizerBase, mlm_prob=0.15):
        self.tokenizer = tokenizer
        self.mlm_prob = mlm_prob

    def __call__(self, examples):
        labels = []
        mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        for e in examples:
            mask_ids = self.get_span_mask_ids(len(e['input_ids']))
            label = [-100] * len(e['input_ids'])
            for idx in mask_ids:
                label[idx] = e['input_ids'][idx]
                e['input_ids'][idx] = mask_id
                labels.append(label)
        inputs = self.tokenizer.pad(examples, return_tensors='pt', padding=True)
        bsz, sql = inputs['input_ids'].shape
        labels = [l + [-100] * (sql - len(l)) for l in labels]
        inputs['labels'] = torch.tensor(np.array(labels))
        return inputs

    def get_span_mask_ids(self, length):
        length -= 1
        if length < 4:
            return set()
        mask_ids = set()
        num_mask = length * self.mlm_prob
        while len(mask_ids) < num_mask:
            p = random.randint(1, length - 1)
            # l = random.randint(2, 6) if random.random() < 0.4 else 1
            l = random.choices([1, 2, 3, 4], [0.4, 0.3, 0.2, 0.1])[0]
            if p + l - 1 > length:
                mask_ids.update(set(range(p, length)))
            else:
                mask_ids.update(set(range(p, p + l - 1)))
        rst = sorted(list(mask_ids))
        assert rst[-1] < length
        return rst
