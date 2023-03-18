import torch
import numpy as np
from typing import Dict, Any, List

from .utilities import gather_batch, convert_word_from_singular_to_plural


class DataCollator:
    def __init__(self, convert_singular_to_plural:bool=False, plural_prefix: str="all_"):
        self.convert_singular_to_plural = convert_singular_to_plural
        self.plural_prefix = plural_prefix

    def apply(self, batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        return batch
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        # gathering
        gathered_batch = gather_batch(batch)
        
        # applying
        try:
            gathered_batch = super().apply(gathered_batch)
        except AttributeError:
            pass

        gathered_batch = self.apply(gathered_batch)

        # setting data types
        example_sample = batch[0]
        for key, value in example_sample.items():
            if key in gathered_batch:
                if not isinstance(value, str) and value is not None:
                    values = gathered_batch[key]
                    if isinstance(value, torch.Tensor):
                        values = torch.stack(values)
                    elif isinstance(value, np.ndarray):
                        values = torch.stack(np.stack(values))

                    gathered_batch[key] = torch.tensor(values)

        # converting singular to plural form
        if self.convert_singular_to_plural:
            for key in gathered_batch.keys():
                key_plural_form = convert_word_from_singular_to_plural(
                    word=key, 
                    plural_prefix=self.plural_prefix,
                )
                gathered_batch[key_plural_form] = gathered_batch.pop(key)
    

        return gathered_batch