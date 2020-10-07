import os
import sentencepiece as spm
from ncc.utils.util_file import load_yaml


def spm_train(input: str, model_prefix: str, vocab_size: int, character_coverage=0.9995, model_type='unigram'):
    command = f"--input={input} --model_prefix={model_prefix} --vocab_size={vocab_size} " \
              f"--character_coverage={character_coverage} --model_type={model_type} --pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3" \
              f" --unk_piece=[UNK] --pad_piece=[PAD] --user_defined_symbols=[LF],[CR] --hard_vocab_limit=false"
    print(command)
    spm.SentencePieceTrainer.Train(command)


if __name__ == '__main__':
    yaml_file = os.path.join(os.path.dirname(__file__), 'share.yml')
    args = load_yaml(yaml_file)

    for modality in args['preprocess']['source_lang']:
        corpus_files = [
            files + f'.{modality}'
            for mode_files in args['preprocess']['dataprefs'].values()
            for mode, files in mode_files.items() if 'train' in mode or 'valid' in mode
        ]
        corpus_file = ','.join(corpus_files)
        model_prefix = os.path.join(args['preprocess']['destdir'], modality)
        os.makedirs(os.path.dirname(model_prefix), exist_ok=True)
        spm_train(corpus_file, model_prefix=model_prefix, vocab_size=50000, model_type='unigram')
