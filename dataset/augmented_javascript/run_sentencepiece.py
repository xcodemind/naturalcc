import argparse
import sentencepiece as spm
import tqdm
import os
from dataset.augmented_javascript.utils.jsonl_dataset import JSONLinesDataset, normalize_docstring
from dataset.augmented_javascript.utils.util import normalize_program
from dataset.codesearchnet.utils.codebert_utils import vocab2dict
# DEFAULT_INPUT = "data/codesearchnet_javascript/javascript_dedupe_definitions_nonoverlap_v2_train.jsonl"
# DEFAULT_OUTPUT = "data/codesearchnet_javascript/javascript_dedupe_definitions_nonoverlap_v2_train.txt"


def make_corpus(input, output):
    dataset = JSONLinesDataset(input, {"function": "function", "docstring": "docstring"})
    print("Number of functions:", len(dataset))
    print("Example original:", dataset[0]["function"])
    print("Example normalized:", normalize_program(dataset[0]["function"]))
    print("Example normalized docstring:", normalize_docstring(dataset[0]["docstring"]))

    with open(output, "w") as f:
        for ex in tqdm.tqdm(dataset, "Writing corpus to txt"):
            # Write docstring
            if ex["docstring"]:
                f.write(normalize_docstring(ex["docstring"]) + "\n")
            # Write normalized function
            function = ex["function"]
            line = normalize_program(function)
            f.write(line + "\n")

    print("Wrote corpus to:", output)


def spm_train(
    input: str, model_prefix: str, vocab_size: int, character_coverage=1.0, model_type='unigram'
):  # , input_sentence_size: int, shuffle_input_sentence: str):
    # command = f"--input={input} --model_prefix={model_prefix} --vocab_size={vocab_size} --character_coverage={character_coverage} --model_type={model_type} --input_sentence_size={input_sentence_size} --shuffle_input_sentence={shuffle_input_sentence}"
    command = f"--input={input} --model_prefix={model_prefix} --vocab_size={vocab_size} " \
              f"--character_coverage={character_coverage} --model_type={model_type} --unk_piece=[UNK] " \
              f"--pad_piece=[PAD] --user_defined_symbols=[CLS],[SEP],[MASK],[EOL],[URL] --hard_vocab_limit=false"
    print(command)
    spm.SentencePieceTrainer.Train(command)


if __name__ == "__main__":
    # fire.Fire({"make_corpus": make_corpus, "spm_train": spm_train})
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", type=str, default='piece', help='id(num)/piece(str)')
    parser.add_argument("--vocab-size", type=int, default=8000, help='token dictionary size')
    parser.add_argument("--src-dir", type=str, default='~/.ncc/augmented_javascript/raw', help='source data')
    parser.add_argument("--tgt-dir", type=str, default='~/.ncc/augmented_javascript/contracode/data-raw', help='save dir for sentencepiece bpe models or save files')
    parser.add_argument("--model-type", type=str, default='bpe',  help='source data')
    parser.add_argument("--model-prefix", type=str, default='csnjs_8k_9995p_unigram_url',  help='source data')

    # parser.add_argument("--bpe-dir", type=str, default='wordpiece_bpe', help='wordpiece_bpe modal save direction')
    parser.add_argument("--keep-empty", type=bool, default=True, help="keep empty lines")
    parser.add_argument("--overwrite", type=bool, default=False, help="build BPE model for files")
    # parser.add_argument("--insert", type=bool, help='insert CLS/S_SEP')
    parser.add_argument("--workers", type=int, default=16, help='multi-processors number')
    args = parser.parse_args()

    args.src_dir = os.path.expanduser(args.src_dir)
    args.tgt_dir = os.path.expanduser(args.tgt_dir)

    input = os.path.join(args.src_dir, 'javascript_dedupe_definitions_nonoverlap_v2_train.jsonl')
    output = os.path.join(args.src_dir, 'javascript_dedupe_definitions_nonoverlap_v2_train.txt')
    # 1. make corpus
    make_corpus(input, output)
    # exit()
    # 2. spm_train
    model_prefix = os.path.join(args.tgt_dir, args.model_prefix)
    spm_train(output, model_prefix=model_prefix, vocab_size=args.vocab_size, model_type=args.model_type)
    vocab2dict(vocab_file='{}.vocab'.format(model_prefix))