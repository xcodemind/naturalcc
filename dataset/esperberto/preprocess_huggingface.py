from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

# paths = [str(x) for x in Path("/data/wanyao/ghproj_d/transformers/eo_data/").glob("**/*.txt")]
paths = [str(x) for x in Path("/data/wanyao/ghproj_d/naturalcodev3/wikitext-103-raw/").glob("**/*.raw")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save files to disk
# tokenizer.save("/data/wanyao/ghproj_d/transformers/eo_data", "esperberto")
tokenizer.save('/data/wanyao/ghproj_d/transformers/models/WikiBERT-small', 'wikibert')