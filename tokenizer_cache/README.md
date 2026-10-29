# tokenizer_cache

The `cl100k_base` and `o200k_base` BPE encodings used by OpenAI's
[tiktoken](https://github.com/openai/tiktoken) (MIT License), stored under the file
names tiktoken expects. They are bundled so token counts work offline and never depend
on a download. `hashedcontext/tokens.py` points tiktoken at this folder automatically.
