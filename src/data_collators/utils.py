import torch
import numpy as np
from typing import Any, List, Optional, Dict, Union
import re


def pad_sequence(
    sequence: Any, 
    max_length: int, 
    padding_value: int = -1, 
    padding_side: str = "right",
    pad_to_multiple_of: Optional[int] = None,
) -> torch.Tensor:
    
    sequence = to_list(sequence)
    sequence_length = len(sequence)
    length_diff = max_length - sequence_length

    if pad_to_multiple_of is not None:
        length_diff = (length_diff + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of

    padding_values =  [padding_value] * length_diff

    if padding_side == "left":
        padded_sequence = padding_values + sequence
    else:
        padded_sequence = sequence + padding_values
    
    padded_sequence = torch.tensor(padded_sequence)
    
    return padded_sequence
    
def to_list(inputs: Any) -> List[Any]:
    if isinstance(inputs, np.ndarray):
        inputs = inputs.tolist()
    elif isinstance(inputs, torch.Tensor):
        inputs = inputs.to("cpu").detach().tolist()

    return list(inputs)

def gather(
    data: List[Dict[str, Any]], 
    batch_wise:bool=False, 
    set_dtypes: bool = False, 
    ignore_keys: Optional[List[str]] = None,
) -> Dict[str, List[Any]]:
    gathered_data = {}
    for sample in data:
        for key, value in sample.items():
            if key not in gathered_data:
                gathered_data[key] = []
            
            if batch_wise:
                gathered_data[key].extend(value)
            else:
                gathered_data[key].append(value)
    
    if set_dtypes:
        gathered_data = set_dtypes(gathered_data, ignore_keys=ignore_keys)

    return gathered_data

def set_dtypes(
    data: Dict[str, List[Any]], 
    ignore_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    
    if ignore_keys is None:
        ignore_keys = []
    
    for key, values in data.items():
        example_value = values[0]

        does_set_value_dtype = (
            not isinstance(example_value, str) and 
            (example_value is not None) and 
            (key not in ignore_keys)
        )

        if does_set_value_dtype:
            if not isinstance(values, torch.Tensor) and isinstance(example_value, torch.Tensor):
                values = torch.stack(values)
            elif isinstance(example_value, np.ndarray):
                values = torch.stack(np.stack(values))

            if not isinstance(values, torch.Tensor):
                values = torch.tensor(values)

        data[key] = values

    return data

def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.sum(x)

def geometric_distribution(
    lower_bound: int, 
    upper_bound: int, 
    p: float = 0.5, 
    return_values: bool = False,
) -> np.ndarray:
    values = np.array(list(range(lower_bound, upper_bound + 1)))
    distribution = p * (1 - p) ** (values - lower_bound)
    distribution = normalize(distribution)

    if return_values:
        return values, distribution

    return distribution

def get_probability_indices(inputs: torch.Tensor, probability: float) -> torch.Tensor:
     probability_matrix = torch.full(inputs.shape, fill_value=probability, dtype=torch.float)
     indices = torch.bernoulli(probability_matrix).bool()

     return indices

def convert_word_from_singular_to_plural(word: str, plural_prefix: Optional[str] = None) -> str:
    plural_form = word
    if re.search("[sxz]$", word) or re.search("[^aeioudgkprt]h$", word):
        plural_form = re.sub("$", "es", word)
    elif re.search("[aeiou]y$", word):
        plural_form = re.sub("y$", "ies", word)
    elif not word.endswith("s"):
        plural_form = word + "s"
    elif plural_prefix is not None:
        plural_form = f"{plural_prefix}word"

    return plural_form

def prepare_sequence_and_label_for_causal_language_modeling(
    sequence: List[Union[str, int]], 
    context_window: int = 128, 
    bos_token: Union[str, int] = 1,
) -> List[List[Union[str, int]]]:
    samples = []
    for token_index in range(2, len(sequence) - 1):
        start_index = max(token_index - context_window, 0)
        end_index = min(len(sequence) - 1, token_index)

        sequence_ = sequence[start_index:end_index]
        
        if bos_token not in sequence_:
            sequence_ = [bos_token] + sequence_[:-1]
            end_index -= 1
            
        label = sequence[end_index]
        
        sample = (sequence_, label)
        samples.append(sample)
        
    return samples