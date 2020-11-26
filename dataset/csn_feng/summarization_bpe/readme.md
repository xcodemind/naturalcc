

Step 1.
```
python -m dataset.csn_msra.summarization_bpe.run_sentencepiece --src-dir ~/.ncc/CodeSearchNet_MSRA/flatten --tgt-dir ~/.ncc/CodeSearchNet_MSRA/summarization/data-mmap --vocab-size 50000 --model-type bpe --language ruby --model-prefix spm_bpe
```


Step 2. 
```
cd ~/.ncc/CodeSearchNet_MSRA/summarization/data-mmap/ruby
cut -f1 spm_bpe_ruby.vocab | tail -n +5 | sed "s/$/ 100/g" > code_tokens.dict.txt
```


Step 3:

```
python -m dataset.csn_msra.summarization_bpe.preprocess
```
Step3 Bug:

```
Traceback (most recent call last):
  File "/home/yang/wanyao/Dropbox/ghproj-titan/naturalcodev3/dataset/csn_msra/summarization_bpe/preprocess.py", line 222, in <module>
    cli_main()
  File "/home/yang/wanyao/Dropbox/ghproj-titan/naturalcodev3/dataset/csn_msra/summarization_bpe/preprocess.py", line 218, in cli_main
    main(args)
  File "/home/yang/wanyao/Dropbox/ghproj-titan/naturalcodev3/dataset/csn_msra/summarization_bpe/preprocess.py", line 205, in main
    make_all(args['preprocess']['source_lang'], src_dict, src_sp)
  File "/home/yang/wanyao/Dropbox/ghproj-titan/naturalcodev3/dataset/csn_msra/summarization_bpe/preprocess.py", line 194, in make_all
    num_workers=args['preprocess']['workers'])
  File "/home/yang/wanyao/Dropbox/ghproj-titan/naturalcodev3/dataset/csn_msra/summarization_bpe/preprocess.py", line 189, in make_dataset
    make_binary_dataset(vocab, in_file, out_file, lang, num_workers)
  File "/home/yang/wanyao/Dropbox/ghproj-titan/naturalcodev3/dataset/csn_msra/summarization_bpe/preprocess.py", line 153, in make_binary_dataset
    input_file, dict, lambda t: ds.add_item(t), offset=0, end=offsets[1]
  File "/home/yang/wanyao/Dropbox/ghproj-titan/naturalcodev3/ncc/data/tools/binarizer.py", line 117, in binarize_bpe
    ids = dict.encode_as_ids(line)  # text => ids
AttributeError: 'Dictionary' object has no attribute 'encode_as_ids'
```
