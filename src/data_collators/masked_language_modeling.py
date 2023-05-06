from transformers import PreTrainedTokenizer
import torch
import copy
from typing import List, Any, Dict

from .dynamic_padding import DynamicPaddingDataCollator
from .utils import get_probability_indices


class MaskedLanguageModelingDataCollator(DynamicPaddingDataCollator):
    def __init__(
        self, 
        input_key: str, 
        label_key: str, 
        tokenizer: PreTrainedTokenizer, 
        special_tokens_mask_key: str = "special_tokens_mask", 
        ignore_index: int = -100, 
        masking_probability: float = 0.15, 
        **args,
    ):
        super().__init__(**args)
        
        self.input_key = input_key
        self.label_key = label_key
        self.tokenizer = tokenizer
        self.special_tokens_mask_key = special_tokens_mask_key
        self.ignore_index = ignore_index
        self.masking_probability = masking_probability
        
        if not (0.0 <= masking_probability <= 1.0):
            raise ValueError(f"`masking_probability`")
    
    def collate(self, batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        inputs = batch[self.input_key]
        labels = copy.deepcopy(inputs)

        special_tokens_mask = batch[self.special_tokens_mask_key].bool()

        # selecting tokens to mask
        selected_indices = get_probability_indices(inputs, probability=self.masking_probability)
        masked_indices = selected_indices & ~special_tokens_mask

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