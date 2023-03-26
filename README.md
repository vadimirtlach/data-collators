# data-collators

Data Collators is a Python library for collating batches of data for Deep Learning projects using PyTorch framework. The library provides a flexible and easy-to-use API for creating custom data collators, which can increase the efficiency of your Deep Learning projects by allowing you to spend more time focusing on your specific problem instead of writing boilerplate code for collating batches for training and inference stages respectively.

Collating data into batches is a critical part of the training and inference processes in Deep Learning tasks. It involves grouping together multiple samples into a single batch, which can then be processed in parallel by the model during training or inference.

With Data Collators, you can easily create custom data collators to suit your specific needs. This allows you to focus on the more important aspects of your Deep Learning project, such as data processing, model architecture and hyperparameter tuning, rather than spending time on the details of data collation.

In addition to the flexible API for creating custom data collators, the library also provides a set of pre-implemented data collators for some Deep Learning tasks. You can find more information about pre-implemented data-collators in the "Pre-implemented Data Collators" section.


### Installation
To install the data-collators library, you should clone the GitHub repository and run the setup script:

```
git clone https://github.com/vad13irt/data-collators.git
cd data-collators
python setup.py install
```

This will install the data-collators library on your machine, allowing you to use it in your projects.

### Usage

Using Data Collators is similar to using the basic PyTorch `collate_fn` API. However, the library provides additional flexibility and convenience by abstracting away many of the low-level details involved in collating data into batches.

```py
from torch.utils.data import DataLoader
from data_collators import DataCollator


# defining dataset
dataset = ...

# defining data loader
collator = DataCollator(convert_singular_to_plural=True)
dataloader = DataLoader(
    dataset=dataset, 
    batch_size=16,
    shuffle=True,
    collate_fn=self.collate_fn, 
)

```

### Creating custom data collators

The Data Collators library provides a flexible and easy-to-use API for creating custom data collators. To create a custom data collator, you simply need to subclass the `DataCollator` class and implement the `apply` method. For example:

```py
from sentence_transformers import SentenceTransformer
from data_collators import DataCollator


class SDIPDataCollator(DataCollator):
    def __init__(
        self, 
        embedding_model_name_or_path="sentence-transformers/all-MiniLM-L6-v2", 
        device="cpu", 
        **args,
    ):
        super().__init__(**args)
        
        self.embedding_model = SentenceTransformer(embedding_model_name_or_path, device=device)
    
    def apply(self, batch):
        images = batch["image"]
         
        if "prompt" in batch:
            prompts = batch["prompt"]
            prompts_embeddings = self.embedding_model.encode(
                sentences=prompts,
                show_progress_bar=False, 
                convert_to_tensor=True
            )

            batch["embedding"] = prompts_embeddings

        return batch
```

This Data Collator was used in the [Stable Diffusion - Image to Prompts](https://www.kaggle.com/competitions/stable-diffusion-image-to-prompts) competition.

### Pre-implemented data collators

The Data Collators library provides pre-implemented data collators for some Deep Learning tasks.

- `DynamicPaddingDataCollator` is a data collator that dynamically pads sequences within a batch to the maximum length in the batch. This is useful when working with sequences of varying lengths, as it ensures that all sequences in the batch have the same length, which is required for processing with deep learning models.

- `MaskedLanguageModelingDataCollator` is a data collator that creates masked language modeling training examples from a batch of input sequences. It randomly masks out tokens in the input sequences and replaces them with a special token, which the model is then trained to predict.

- `ReplacedTokenDetectionDataCollator` is a data collator that creates replaced token detection training examples from a batch of input sequences. It randomly replaces tokens in the input sequences with a special token, and the model is trained to detect which tokens were replaced.

- `SpanMaskingDataCollator` is a data collator that creates span masking training examples from a batch of input sequences. It randomly masks out a contiguous span of tokens in the input sequences and replaces them with a special token, which the model is then trained to predict.


### Contributing
We welcome and encourage contributions to Data Collators! Whether it's a bug fix, a new feature, or an improvement to documentation, all contributions are appreciated. Don't be shy - let's make Data Collators even better together!

### License
This repository is licensed under the Apache 2.0 License, which means that you are free to use, modify, and distribute the code as long as you adhere to the terms and conditions of the license. Please see the LICENSE file for more details.