from transformers import PreTrainedTokenizer
import torch
import copy
from typing import Dict, List, Any

from .dynamic_padding import DynamicPaddingDataCollator
from .utils import get_probability_indices


class ReplacedTokenDetectionDataCollator(DynamicPaddingDataCollator):
    def __init__(
        self, 
        input_key: str, 
        label_key: str, 
        tokenizer: PreTrainedTokenizer, 
        special_tokens_mask_key: str = "special_tokens_mask", 
        replace_token_probability: float = 0.15, 
        **args,
    ):
        super().__init__(**args)
        
        self.input_key = input_key
        self.label_key = label_key
        self.tokenizer = tokenizer
        self.special_tokens_mask_key = special_tokens_mask_key
        self.replace_token_probability = replace_token_probability
        
        if not (0.0 <= replace_token_probability <= 1.0):
            raise ValueError(f"`replace_token_probability`")
            
    def apply(self, batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:        
        inputs = batch[self.input_key]
        labels = copy.deepcopy(inputs)

        special_tokens_mask = batch[self.special_tokens_mask_key].bool()
        
        # replacing tokens with random token from vocabulary
        random_token_indices = get_probability_indices(inputs, probability=self.replace_token_probability) & ~special_tokens_mask
        inputs[random_token_indices] = torch.randint(len(self.tokenizer), inputs.shape, dtype=torch.long)

        # creating label
        labels = (inputs != labels).int()
        
        batch.update({
            self.input_key: inputs,
            self.label_key: labels,
        })
        
        return batch