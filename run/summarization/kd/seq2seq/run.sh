#!/usr/bin/env bash

# Knowledge Distillation
# 1) train {teacher networks} on language1, language2, ...
# 2) generate validation bleu score, train dataset topk probabilities/indices of {teacher networks}
# 3) distill some {teacher networks} into a new student network
# 4) finetune such a student network on target dataset

# Examples:
# In our scenario, we want to utilize csn_feng dataset, including python/php/java/javascript/go/ruby, to improve the performance on python_wan/java_hu
# 1) train teacher networks on python/php/java/javascript/go/ruby datasets
# 2) generate validation bleu score, train dataset topk probabilities/indices of those {teacher networks}
# 3) distill teacher networks into a new student network
# 4) finetune such a student network on target (python_wan/java_hu) dataset

# =================================== Teacher(ruby) =================================== #
# one teacher - ruby
CUDA_VISIBLE_DEVICES=1,2,3 nohup python -m run.summarization.kd.seq2seq.train_teacher -l ruby > run/summarization/kd/seq2seq/teacher.ruby.log 2>&1 &
# watch -n 2 "tail -n 20 run/summarization/kd/seq2seq/teacher.ruby.log"

# eval teacher
CUDA_VISIBLE_DEVICES=3 python -m run.summarization.kd.seq2seq.eval -l teacher_generate/ruby

# generate teacher topk
CUDA_VISIBLE_DEVICES=3 python -m run.summarization.kd.seq2seq.teacher_generate -l ruby

# {'bleu_ruby': 4.7186275146}
# [2020-10-06 20:31:43# Bleu.Json: /home/wanyao/.ncc/kd/code_tokens_docstring_tokens/data-mmap/expert_bleu_ruby_code_tokens_docstring_tokens.json,
# TopK.Idx/Prob: /home/wanyao/.ncc/kd/code_tokens_docstring_tokens/data-mmap/ruby_code_tokens_docstring_tokens_topk_prob.

# {'bleu_python': 5.749516218155357}
# [2020-10-06 20:31:43# Bleu.Json: /home/wanyao/.ncc/kd/code_tokens_docstring_tokens/data-mmap/expert_bleu_python_code_tokens_docstring_tokens.json,
# TopK.Idx/Prob: /home/wanyao/.ncc/kd/code_tokens_docstring_tokens/data-mmap/python_code_tokens_docstring_tokens_topk_prob.

# {'bleu_php': 5.934669357751767}
# [2020-10-06 20:31:43# Bleu.Json: /home/wanyao/.ncc/kd/code_tokens_docstring_tokens/data-mmap/expert_bleu_php_code_tokens_docstring_tokens.json,
# TopK.Idx/Prob: /home/wanyao/.ncc/kd/code_tokens_docstring_tokens/data-mmap/php_code_tokens_docstring_tokens_topk_prob.

# {'bleu_java': 5.802408973533259}
# [2020-10-06 20:31:43# Bleu.Json: /home/wanyao/.ncc/kd/code_tokens_docstring_tokens/data-mmap/expert_bleu_java_code_tokens_docstring_tokens.json,
# TopK.Idx/Prob: /home/wanyao/.ncc/kd/code_tokens_docstring_tokens/data-mmap/java_code_tokens_docstring_tokens_topk_prob.

# {'bleu_javascript': 4.906121226126045}
# [2020-10-06 20:31:43# Bleu.Json: /home/wanyao/.ncc/kd/code_tokens_docstring_tokens/data-mmap/expert_bleu_javascript_code_tokens_docstring_tokens.json,
# TopK.Idx/Prob: /home/wanyao/.ncc/kd/code_tokens_docstring_tokens/data-mmap/javascript_code_tokens_docstring_tokens_topk_prob.

# {'bleu_go': 8.253784917935445}
# [2020-10-06 20:31:43# Bleu.Json: /home/wanyao/.ncc/kd/code_tokens_docstring_tokens/data-mmap/expert_bleu_go_code_tokens_docstring_tokens.json,
# TopK.Idx/Prob: /home/wanyao/.ncc/kd/code_tokens_docstring_tokens/data-mmap/go_code_tokens_docstring_tokens_topk_prob.



# =================================== Teacher(python) =================================== #
# one teacher - python
CUDA_VISIBLE_DEVICES=1,2,3 nohup python -m run.summarization.kd.seq2seq.train_teacher -l python > run/summarization/kd/seq2seq/teacher.python.log 2>&1 &
# watch -n 2 "tail -n 20 run/summarization/kd/seq2seq/teacher.python.log"

# eval teacher
CUDA_VISIBLE_DEVICES=1 python -m run.summarization.kd.seq2seq.eval -l teacher_generate/python

# generate teacher topk
CUDA_VISIBLE_DEVICES=3 python -m run.summarization.kd.seq2seq.teacher_generate -l python
# {'bleu_python': }
# [2020-10-06 20:31:43# Bleu.Json: /home/wanyao/.ncc/kd/code_tokens_docstring_tokens/data-mmap/expert_bleu_python_code_tokens_docstring_tokens.json,
# TopK.Idx/Prob: /home/wanyao/.ncc/kd/code_tokens_docstring_tokens/data-mmap/python_code_tokens_docstring_tokens_topk_prob. (fed_utils.py:371, save_expert_outputs())


# =================================== Teacher(XX) =================================== #
# one teacher - XX
CUDA_VISIBLE_DEVICES=1,2,3 nohup python -m run.summarization.kd.seq2seq.train_teacher -l XX > run/summarization/kd/seq2seq/teacher.XX.log 2>&1 &
# watch -n 2 "tail -n 20 run/summarization/kd/seq2seq/teacher.XX.log"

# eval teacher
CUDA_VISIBLE_DEVICES=3 python -m run.summarization.kd.seq2seq.eval -l teacher_generate/XX

# generate teacher topk
CUDA_VISIBLE_DEVICES=3 python -m run.summarization.kd.seq2seq.teacher_generate -l XX
# {'bleu_XX': }
# [2020-10-06 20:31:43# Bleu.Json: /home/wanyao/.ncc/kd/code_tokens_docstring_tokens/data-mmap/expert_bleu_XX_code_tokens_docstring_tokens.json,
# TopK.Idx/Prob: /home/wanyao/.ncc/kd/code_tokens_docstring_tokens/data-mmap/XX_code_tokens_docstring_tokens_topk_prob. (fed_utils.py:371, save_expert_outputs())

# =================================== Teacher(XX) =================================== #
# one teacher - XX
CUDA_VISIBLE_DEVICES=1,2,3 nohup python -m run.summarization.kd.seq2seq.train_teacher -l XX > run/summarization/kd/seq2seq/teacher.XX.log 2>&1 &
# watch -n 2 "tail -n 20 run/summarization/kd/seq2seq/teacher.XX.log"

# eval teacher
CUDA_VISIBLE_DEVICES=3 python -m run.summarization.kd.seq2seq.eval -l teacher_generate/XX

# generate teacher topk
CUDA_VISIBLE_DEVICES=3 python -m run.summarization.kd.seq2seq.teacher_generate -l XX
# {'bleu_XX': }
# [2020-10-06 20:31:43# Bleu.Json: /home/wanyao/.ncc/kd/code_tokens_docstring_tokens/data-mmap/expert_bleu_XX_code_tokens_docstring_tokens.json,
# TopK.Idx/Prob: /home/wanyao/.ncc/kd/code_tokens_docstring_tokens/data-mmap/XX_code_tokens_docstring_tokens_topk_prob. (fed_utils.py:371, save_expert_outputs())

# =================================== Teacher(XX) =================================== #
# one teacher - XX
CUDA_VISIBLE_DEVICES=1,2,3 nohup python -m run.summarization.kd.seq2seq.train_teacher -l XX > run/summarization/kd/seq2seq/teacher.XX.log 2>&1 &
# watch -n 2 "tail -n 20 run/summarization/kd/seq2seq/teacher.XX.log"

# eval teacher
CUDA_VISIBLE_DEVICES=3 python -m run.summarization.kd.seq2seq.eval -l teacher_generate/XX

# generate teacher topk
CUDA_VISIBLE_DEVICES=3 python -m run.summarization.kd.seq2seq.teacher_generate -l XX
# {'bleu_XX': }
# [2020-10-06 20:31:43# Bleu.Json: /home/wanyao/.ncc/kd/code_tokens_docstring_tokens/data-mmap/expert_bleu_XX_code_tokens_docstring_tokens.json,
# TopK.Idx/Prob: /home/wanyao/.ncc/kd/code_tokens_docstring_tokens/data-mmap/XX_code_tokens_docstring_tokens_topk_prob. (fed_utils.py:371, save_expert_outputs())

# =================================== Teacher(XX) =================================== #
# one teacher - XX
CUDA_VISIBLE_DEVICES=1,2,3 nohup python -m run.summarization.kd.seq2seq.train_teacher -l XX > run/summarization/kd/seq2seq/teacher.XX.log 2>&1 &
# watch -n 2 "tail -n 20 run/summarization/kd/seq2seq/teacher.XX.log"

# eval teacher
CUDA_VISIBLE_DEVICES=3 python -m run.summarization.kd.seq2seq.eval -l teacher_generate/XX

# generate teacher topk
CUDA_VISIBLE_DEVICES=3 python -m run.summarization.kd.seq2seq.teacher_generate -l XX
# {'bleu_XX': }
# [2020-10-06 20:31:43# Bleu.Json: /home/wanyao/.ncc/kd/code_tokens_docstring_tokens/data-mmap/expert_bleu_XX_code_tokens_docstring_tokens.json,
# TopK.Idx/Prob: /home/wanyao/.ncc/kd/code_tokens_docstring_tokens/data-mmap/XX_code_tokens_docstring_tokens_topk_prob. (fed_utils.py:371, save_expert_outputs())



# =================================== Distillation(ruby2python_wan) =================================== #
# use valid_bleu, topk idx and prob to distill
CUDA_VISIBLE_DEVICES=3 nohup python -m run.summarization.kd.seq2seq.train_student -l ruby > run/summarization/kd/seq2seq/student.ruby.log 2>&1 &
# watch -n 2 "tail -n 20 run/summarization/kd/seq2seq/student.ruby2python_wan.log"

CUDA_VISIBLE_DEVICES=1,2,3 nohup python -m run.summarization.kd.seq2seq.train_student -l csn > run/summarization/kd/seq2seq/student.csn.log 2>&1 &
# watch -n 2 "tail -n 20 run/summarization/kd/seq2seq/student.csn.log"

# =================================== Finetune(python_wan) =================================== #

CUDA_VISIBLE_DEVICES=1,2,3 nohup python -m run.summarization.kd.seq2seq.finetune -l python_wan > run/summarization/kd/seq2seq/finetune.csn.python_wan.log 2>&1 &
# watch -n 2 "tail -n 20 run/summarization/kd/seq2seq/finetune.csn.python_wan.log"