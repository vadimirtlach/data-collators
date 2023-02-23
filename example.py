import transformers
from torch.utils.data import Dataset, DataLoader
from data_collators import ReplacedTokenDetectionDataCollator
from datasets import load_dataset
import pprint


class PretrainDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        super().__init__()

        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = self.texts[index]

        tokenized = self.tokenizer(
            text=text, 
            max_length=self.max_length, 
            padding=False, 
            truncation=True,  
            return_attention_mask=True, 
            return_special_tokens_mask=True,
            return_token_type_ids=False,
        )

        tokenized.update({
            "text": text,
        })

        return tokenized
    

# data
dataset_name = "rotten_tomatoes"
dataset = load_dataset(dataset_name, split="train")

# config
model_name_or_path = "bert-base-uncased"
max_length = 512
batch_size = 4

# tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
max_length = tokenizer.model_max_length

# dataset and data loader
pretrain_dataset = PretrainDataset(
    texts=dataset["text"], 
    tokenizer=tokenizer, 
    max_length=max_length,
)

data_collator = ReplacedTokenDetectionDataCollator(
    input_key="input_ids",
    label_key="label",
    tokenizer=tokenizer,
    replace_token_probability=0.15,
    padding_keys=["input_ids", "attention_mask", "special_tokens_mask"], 
    padding_values=[tokenizer.pad_token_id, 1, 1],
)

dataloader = DataLoader(
    dataset=pretrain_dataset, 
    batch_size=batch_size, 
    collate_fn=data_collator,
)

# results
index = 0
sample = dataset[index]
batch = next(iter(dataloader))

print("Sample")
print(sample)
print("\n"*3)
print("Batch")
print("Inputs:", batch["input_ids"])
print("Labels:", batch["label"])


# Sample
# {'text': 'the rock is destined to be the 21st century\'s new " conan " and that he\'s going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .', 'label': 1}




# Batch
# Inputs: tensor([[  101,  1996,  2600,  2003, 16036,  2000,  2022,  1996,  7398,  2301,
#           1005,  1055,  2047,  1000, 16608,  1000,  1998,  2008,  2002,  1005,
#           1055,  2183,  2000,  2191,  1037, 17624,  2130,  3618,  2084,  7779,
#          29058,  8625, 13327,  1010,  3744,  1011, 18856, 19513,  3158,  5477,
#           4168,  2030,  7112, 16562,  2140,  1012,   102,     0,     0,     0,
#              0,     0],
#         [  101,  1996,  9882,  2135,  9603, 13633,  1997,  1000,  1996,  2935,
#           1997,  1996,  7635,  1000, 11544,  2003,  2061,  4121,  2008,  1037,
#           5930,  1997,  2616,  3685, 23613,  6235,  2522,  1011,  3213,  1013,
#           2472,  2848,  4027,  1005,  1055,  4423,  4432,  1997,  1046,  1012,
#           1054,  1012,  1054,  1012, 23602,  1005,  1055,  2690,  1011,  3011,
#           1012,   102],
#         [  101,  4621,  2021,  2205,  1011,  8915, 23267, 16012, 24330,   102,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0],
#         [  101,  2065,  2017,  2823,  2066,  2000,  2175,  2000,  1996,  5691,
#           2000,  2031,  4569,  1010,  2001, 28518,  2003,  1037,  2204,  2173,
#           2000,  2707,  1012,   102,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0]])
# Labels: tensor([[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
#          0, 0, 0, 0],
#         [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
#          0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
#          0, 0, 1, 0],
#         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0],
#         [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0]], dtype=torch.int32)