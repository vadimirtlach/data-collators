from transformers import PreTrainedTokenizer
import numpy as np
import torch
import math
import copy
from typing import Dict, List, Any

from .dynamic_padding import DynamicPaddingDataCollator
from .utils import geometric_distribution, get_probability_indices


class SpanMaskingDataCollator(DynamicPaddingDataCollator):
    """
    Span Masking - https://arxiv.org/abs/1907.10529

    References:
        https://github.com/rbiswasfc/kaggle-feedback-effectiveness-3rd-place-solution/blob/main/code/tools/fpe_span_mlm.py
    """
    
    def __init__(
        self, 
        input_key: str, 
        label_key: str, 
        tokenizer: PreTrainedTokenizer, 
        ignore_index: int = -100, 
        min_span_length: int = 1, 
        max_span_length: int = 10, 
        special_tokens_mask_key: str = "special_tokens_mask", 
        span_masking_probability: float = 0.15, 
        length_probability: float = 0.2, 
        **args,
    ):
        super().__init__(**args)
        
        self.input_key = input_key
        self.label_key = label_key
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.special_tokens_mask_key = special_tokens_mask_key
        self.span_masking_probability = span_masking_probability
        self.min_span_length = min_span_length
        self.max_span_length = max_span_length
        self.length_probability = length_probability
        
        if not (0.0 <= span_masking_probability <= 1.0):
            raise ValueError(f"`span_masking_probability`")

    def apply(self, batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        inputs = batch[self.input_key]
        labels = copy.deepcopy(inputs)

        special_tokens_masks = batch[self.special_tokens_mask_key]

        masked_indices = []
        for label, special_tokens_mask in zip(labels, special_tokens_masks):
            num_tokens = len(label)
            num_masked_tokens = math.ceil(num_tokens * self.span_masking_probability)
            
            indices = list(range(len(label)))
            
            max_span_length = self.max_span_length
            if max_span_length is None:
                max_span_length = torch.sum((1 - special_tokens_mask), axis=-1)
                
            lengths, probabilities = geometric_distribution(
                lower_bound=self.min_span_length, 
                upper_bound=max_span_length, 
                p=self.length_probability, 
                return_values=True,
            )
            
            label_masked_indices = []
            while len(label_masked_indices) <= num_masked_tokens:
                start_indice = np.random.choice(indices)
                span_length = np.random.choice(lengths, p=probabilities)
                end_indice = min(start_indice + span_length, max_span_length + 1)
                
                if start_indice in label_masked_indices:
                    continue
                
                for indice in range(start_indice, end_indice):
                    if len(label_masked_indices) <= num_masked_tokens:
                        label_masked_indices.append(indice)
            
            mask = torch.tensor([
                token_index in label_masked_indices for token_index in range(num_tokens)
            ]).bool()
            
            non_special_tokens_mask = mask & ~special_tokens_mask
            masked_indices.append(non_special_tokens_mask)
        
        masked_indices = torch.stack(masked_indices).bool()

        # replacing tokens with [MASK] token
        mask_token_indices = get_probability_indices(inputs, probability=0.8) & masked_indices
        inputs[mask_token_indices] = self.tokenizer.mask_token_id

        # replacing tokens with random token from vocabulary
        random_token_indices = get_probability_indices(inputs, probability=0.5) & masked_indices & ~mask_token_indices
        inputs[random_token_indices] = torch.randint(len(self.tokenizer), inputs.shape, dtype=torch.long)
        
        # creating label
        labels[~masked_indices] = self.ignore_index

        
        batch.update({
            self.input_key: inputs,
            self.label_key: labels,
        })

        return batch