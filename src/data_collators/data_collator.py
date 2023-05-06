from typing import Dict, Any, List, Optional

from .utils import (
    gather, 
    set_dtypes, 
    convert_word_from_singular_to_plural,
)


class DataCollator:
    def __init__(
        self, 
        ignore_keys:Optional[List[str]] = [], 
        convert_singular_to_plural:bool = False, 
        plural_prefix: str = "all_",
    ):
        self.convert_singular_to_plural = convert_singular_to_plural
        self.plural_prefix = plural_prefix
        self.ignore_keys = ignore_keys

        if self.ignore_keys is None:
            self.ignore_keys = []

    def collate(self, batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        return batch
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        # gathering
        batch = gather(batch, set_dtypes=False)
        
        # collating
        try:
            batch = super().collate(batch)
        except AttributeError:
            pass

        batch = self.collate(batch)

        # setting data types
        batch = set_dtypes(batch, ignore_keys=self.ignore_keys)

        # converting singular to plural form
        if self.convert_singular_to_plural:
            keys = list(batch.keys())
            for key in keys:
                if key not in self.ignore_keys:
                    key_plural_form = convert_word_from_singular_to_plural(
                        word=key, 
                        plural_prefix=self.plural_prefix,
                    )
                    batch[key_plural_form] = batch.pop(key)
    

        return batch