import torch
from typing import List, Optional, Dict, Any, Union

from .data_collator import DataCollator
from .utils import pad_sequence


class DynamicPaddingDataCollator(DataCollator):    
    def __init__(
        self, 
        mapping: Dict[str, Any],
        max_length: Optional[Union[str, int]] = None,
        padding_side: str = "right", 
        pad_to_multiple_of: Optional[int] = None,
        ignore_missing_keys: bool = True,
        **args,
    ):
        super().__init__(**args)

        self.mapping = mapping
        self.max_length = max_length
        self.padding_side = padding_side
        self.pad_to_multiple_of = pad_to_multiple_of
        self.ignore_missing_keys = ignore_missing_keys

        
    def get_max_length(self, sequences):
        lengths = [len(sequence) for sequence in sequences]
        return max(lengths)

    def collate(self, batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:                        
        if self.ignore_missing_keys:
            new_mapping = {}
            for padding_key, padding_value in self.mapping.items():
                if padding_key in batch:
                    new_mapping[padding_key] = padding_value
            
            self.mapping = new_mapping
        
        if self.max_length is not None:
            max_length = (
                self.get_max_length(sequences=batch[self.max_length]) 
                if isinstance(self.max_length, str) 
                else self.max_length
            )
        
        for padding_key, padding_value in self.mapping.items():
            sequences = batch[padding_key]

            if self.max_length is None:
                max_length = (
                    self.get_max_length(sequences) 
                    if isinstance(self.max_length, str) 
                    else self.max_length
                )
        
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