from typing import Dict, Optional, Any

from ..data_collator import DataCollator
from ..dynamic_padding import DynamicPaddingDataCollator


def generalize_labels_key(batch, label_keys=("label", "label_ids")):
    for key in label_keys:
        if key in batch:
            batch["labels"] = batch.pop(key, None)

    return batch

class DefaultDataCollator(DataCollator):
    def collate(self, batch):
        return generalize_labels_key(batch)
    
class HuggingFaceDataCollatorWrapper:
    def __init__(
        self, 
        tokenizer, 
        additional_mapping: Optional[Dict[str, Any]] = None, 
        pad_to_multiple_of=None,
        ignore_missing_keys=False,
        **args,
    ):
        self.tokenizer = tokenizer
        self.additional_mapping = additional_mapping

        if self.additional_mapping is None:
            self.additional_mapping = {}

        self.mapping = {
            "input_ids": self.tokenizer.pad_token_id,
            "attention_mask": 0,
            "token_type_ids": self.tokenizer.pad_token_type_id,
            "offset_mapping": (0, 0),
        }.update(self.additional_mapping)

        self.padding_side = self.tokenizer.padding_side
        self.max_length = "input_ids"
        self.pad_to_multiple_of = pad_to_multiple_of
        self.ignore_missing_keys = ignore_missing_keys

        self.dynamic_padding = DynamicPaddingDataCollator(
            mapping=self.mapping,
            max_length=self.max_length,
            padding_side=self.padding_side,
            pad_to_multiple_of=self.pad_to_multiple_of,
            ignore_missing_keys=self.ignore_missing_keys,
        )

    def default_collate(self, batch):
        batch = self.dynamic_padding(batch)
        batch = generalize_labels_key(batch)

        return batch