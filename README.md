# # GPT2 Pytorch

Extremely simple and understandable GPT2 implementation with minor tweaks.

## Advantages

* You can train even the subword tokenizer, good for non-English languages.
* Fast optimized code, you can train it with a single GTX 2080ti card
* Easy to understand, solid code
* Easy to extend for new experiments

## Supported features
* Lamb optimizer
* Mixed precision training, the important layers remained in fp32.
* sin, cos positional encoding
