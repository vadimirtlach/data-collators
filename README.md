# data-collators

### Masked Language Modeling

1. Select a portion of the tokens.
2. Replace 80% of the selected tokens with a special [MASK] token.
3. Randomly replace 10% of the selected tokens with words from the vocabulary.
4. Leave the last 10% of the selected tokens unchanged.
5. Assign an ignore index value to the unmasked tokens, since the loss will only be computed on the masked tokens.

### Replaced Token Detection

1. Choose a specific portion of the tokens and replace them with randomly selected words from the vocabulary.
2. Create binary labels to indicate whether each token was changed or not, using 0 for not changed and 1 for changed.

### Span Masking
1. A subset of tokens is selected from a given sequence by iteratively sampling spans of text until a certain masking budget (e.g. 15% of the original sequence) has been spent.
2. Masked Language Modeling is applied to the selected subset of tokens.

### TO-DO

#### Tasks

- [ ] Additional parameters for Masked Language Modeling: https://arxiv.org/abs/2202.08005
- [ ] Alternatives for Masked Language Modeling: https://arxiv.org/abs/2109.01819
- [ ] Permutation Language Modeling
- [ ] Whole Word Masking
- [ ] Translation Language Modeling
- [ ] Causal Language Modeling

#### API
- [ ] Access developers to some functionalities (e.g masking, choosing tokens with probability, etc.) through utilities.
- [ ] Make writing custom pre-training techniques more easily. 
- [ ] Compose
- [ ] Apply method


#### Documentation
- [ ] Include documentation strings for both the classes and functions.
- [ ] Demonstrate the usage of data collators for various tasks.