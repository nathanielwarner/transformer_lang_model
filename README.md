# Transformer-based Code Completion

A proof-of-concept source code completion model,
based on the [Universal Transformer](https://arxiv.org/abs/1807.03819) 
architecture, trained on [2.1 million Java methods](http://leclair.tech/data/funcom/).

## Technical Details

In contrast to the original proposed Transformer, which is made up of a set of
layers, each with their own parameters, the Universal Transformer applies the same
layer repeatedly. This improves performance across many tasks, particularly
those of an algorithmic nature (such as processing source code as opposed to 
natural language).

In tasks such as completion where we are not translating between languages,
only the "Encoder" layer of the Transformer is utilized.

The steps to predict the next token, given an input prompt:
1. The prompt is tokenized using a [SentencePiece](https://github.com/google/sentencepiece) model
2. The set of input tokens are processed by an embedding layer, which turns
them into vectors
3. A [Transformer Encoder Layer](https://pytorch.org/docs/master/generated/torch.nn.TransformerEncoderLayer.html) 
is applied *n=8* times
4. Finally, a Dense layer is used to produce next-token probabilities

This process is applied repeatedly until the end-of-sentence token is reached,
or a specified maximum length is surpassed.

## Running Locally
You'll need Python 3 with ``torch`` and ``sentencepiece``.
Run ``run.py`` with Python to enter an interactive demo.

You can train a new model by editing the parameters in ``train.py``.
