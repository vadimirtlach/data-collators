import torch
import numpy as np
from typing import Dict, Any, List

from .utilities import gather_batch


class DataCollator:
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

        return gathered_batch