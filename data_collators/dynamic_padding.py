import torch
from typing import List, Optional, Dict, Any

from .data_collator import DataCollator
from .utilities import pad_sequence


class DynamicPaddingDataCollator(DataCollator):
    """
    TODO:
    auto detection
    """
    
    def __init__(self, 
                 padding_keys: List[str], 
                 padding_values: List[str], 
                 padding_side: str = "right", 
                 pad_to_multiple_of: Optional[int] = None):
        super().__init__()
        self.padding_keys = padding_keys
        self.padding_values = padding_values
        self.padding_side = padding_side
        self.pad_to_multiple_of = pad_to_multiple_of
        
    def apply(self, batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:                
        max_length_key = self.padding_keys[0]
        lengths = [
            len(sequence) for sequence in batch[max_length_key]
        ]
        max_length = max(lengths)
        
        for padding_key, padding_value in zip(self.padding_keys, self.padding_values):
            sequences = batch[padding_key]
            padded_sequences = [
                pad_sequence(
                    sequence=sequence, 
                    max_length=max_length, 
                    padding_value=padding_value, 
                    padding_side=self.padding_side, 
                    pad_to_multiple_of=self.pad_to_multiple_of,
                ) for sequence in sequences
            ]
            
            batch[padding_key] = torch.stack(padded_sequences)
        
        return batch