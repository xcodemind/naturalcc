#!/usr/bin/env bash


# kd_mm2seq
python -u ./run/main.py --yaml student_go_java_javascript_php_python.yml \
--task summarization --lang_mode xlang \
--method_name kd_mm2seq --train_mode train_sl --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 7-8.1 \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

################

# code2seq

python -u ./run/main.py --yaml yml_dir/ruby_code2seq_p1.0_s.yml \
--task summarization --lang_mode unilang \
--method_name code2seq --train_mode train_sl --dataset_type source \
--debug 0 --multi_processing 0  \
--occupy_gpu 0-6    \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml yml_dir/ruby_code2seq_p1.0_s.yml \
--task summarization --lang_mode unilang \
--method_name code2seq --train_mode test --dataset_type source \
--debug 0 --multi_processing 0  \
--occupy_gpu 6-6    \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml yml_dir/ruby_code2seq_p0_s.yml \
--task summarization --lang_mode unilang \
--method_name code2seq --train_mode test --dataset_type source \
--debug 0 --multi_processing 0  \
--occupy_gpu 3-6    \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/


python -u ./run/main.py --yaml yml_dir/ruby_code2seq_p0_s.yml \
--task summarization --lang_mode unilang \
--method_name code2seq --train_mode case_study  --dataset_type source \
--debug 0 --multi_processing 0  \
--occupy_gpu 7-2.2       \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml yml_dir/ruby_code2seq_p1.0_s.yml \
--task summarization --lang_mode unilang \
--method_name code2seq --train_mode case_study  --dataset_type source \
--debug 0 --multi_processing 0  \
--occupy_gpu 7-2.2     \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

# ast_attendgruv4
python -u ./run/main.py --yaml ruby.yml \
--task summarization --lang_mode unilang \
--method_name ast_attendgruv4 --train_mode train_sl --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 7-8.1 \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_p0.2.yml  \
--task summarization --lang_mode unilang  \
--method_name ast_attendgruv4 --train_mode train_sl --dataset_type source \
--debug 0 --multi_processing 0  \
--occupy_gpu 2-2.8          \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_p0.01.yml  \
--task summarization --lang_mode unilang  \
--method_name ast_attendgruv4 --train_mode train_sl --dataset_type source \
--debug 0 --multi_processing 0  \
--occupy_gpu 2-2.8          \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

# codenn
python -u ./run/main.py --yaml ruby.yml \
--task summarization --lang_mode unilang \
--method_name codenn --train_mode train_sl --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 7-8.1 \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_p0.2.yml  \
--task summarization --lang_mode unilang  \
--method_name codenn --train_mode train_sl --dataset_type source  \
--debug 0 --multi_processing 0   \
--occupy_gpu 5-1.6      \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_p0.01.yml  \
--task summarization --lang_mode unilang  \
--method_name codenn --train_mode train_sl --dataset_type source  \
--debug 0 --multi_processing 0   \
--occupy_gpu 5-1.6        \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_p0.2.yml   \
--task summarization --lang_mode unilang  \
--method_name codenn --train_mode test  --dataset_type source  \
--debug 0 --multi_processing 0   \
--occupy_gpu 5-1.6      \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_p0.01.yml   \
--task summarization --lang_mode unilang  \
--method_name codenn --train_mode test  --dataset_type source  \
--debug 0 --multi_processing 0   \
--occupy_gpu 5-1.6        \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

# seq2seq
python -u ./run/main.py --yaml ruby_seq2seq_s.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode train_sl --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 3-4.2  \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_seq2seq_s.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode test --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 3-4.2  \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

# seq2seq v2
python -u ./run/main.py --yaml ruby_seq2seq_v2_p0.01_s.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode train_sl --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 7-4.5    \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_seq2seq_v2_p0.2_s.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode train_sl --dataset_type source \
--debug 0 --multi_processing 0    \
--occupy_gpu 7-4       \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_seq2seq_v2_s.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode train_sl --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 7-4.2   \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_seq2seq_v2_s.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode test  --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 3-5.2     \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

# running
python -u ./run/main.py --yaml ruby_seq2seq_v2_s.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode case_study  --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 7-4      \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/



# running
python -u ./run/main.py --yaml ruby_seq2seq_v2_p0_s.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode case_study  --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 7-2.6        \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

# tree2seq
python -u ./run/main.py --yaml ruby_tree2seq_s.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode train_sl --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 5-2.6  \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_tree2seq_p0.01_s.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode train_sl --dataset_type source \
--debug 0 --multi_processing 0   \
--occupy_gpu 7-2.6     \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_tree2seq_p0.2_s.yml \
--task summarization --lang_mode unilang    \
--method_name mm2seq --train_mode train_sl --dataset_type source   \
--debug 0 --multi_processing 0     \
--occupy_gpu 7-2.6     \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_tree2seq_s.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode test --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 3-2.6  \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

# deepcom
python -u ./run/main.py --yaml ruby_s.yml \
--task summarization --lang_mode unilang \
--method_name deepcom --train_mode train_sl --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 3-5    \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_p0.2_s.yml \
--task summarization --lang_mode unilang \
--method_name deepcom --train_mode train_sl --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 5-5    \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_p0.01_s.yml \
--task summarization --lang_mode unilang \
--method_name deepcom --train_mode train_sl --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 5-2.6       \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_s.yml \
--task summarization --lang_mode unilang   \
--method_name deepcom --train_mode test --dataset_type source \
--debug 0 --multi_processing 0  \
--occupy_gpu 0-5.8    \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

#################################   mm2seq

#  tok8path attn    ruby  240g1
python -u ./run/main.py --yaml ruby_tok8path_attn_s.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode train_sl --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 1-6  \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_tok8path_attn_p0.2_s.yml \
--task summarization --lang_mode unilang  \
--method_name mm2seq --train_mode train_sl --dataset_type source  \
--debug 0 --multi_processing 0  \
--occupy_gpu 5-7      \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_tok8path_attn_p0.01_s.yml \
--task summarization --lang_mode unilang  \
--method_name mm2seq --train_mode train_sl --dataset_type source  \
--debug 0 --multi_processing 0  \
--occupy_gpu 7-7       \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_tok8path_attn_s.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode test --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 6-6    \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

#  tok8path    ruby  243g3
python -u ./run/main.py --yaml ruby_tok8path_s.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode train_sl --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 3-4   \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_tok8path_p0.2_s.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode train_sl --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 2-6   \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_tok8path_p0.01_s.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode train_sl --dataset_type source \
--debug 0 --multi_processing 0   \
--occupy_gpu 5-6   \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_tok8path_s.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode test --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 6-6    \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/


# tok attn ruby 243g0 wanyao
python -u ./run/main.py --yaml ruby_tok_attn_s.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode train_sl --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 0-5.2     \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_tok_attn_s.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode test --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 6-6       \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

# path ruby 243g6  wanyao deployed
python -u ./run/main.py --yaml ruby_path_s.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode train_sl --dataset_type source \
--debug 0 --multi_processing 0  \
--occupy_gpu 6-3.5      \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_path_s.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode test --dataset_type source \
--debug 0 --multi_processing 0  \
--occupy_gpu 6-6     \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

# path attn pointer 243g2 wanyao  还没实现
python -u ./run/main.py --yaml ruby_path_attn_pointer_s.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode train_sl --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 2-5.2     \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/




# tok attn pointer 243g0 wanyao deployed
python -u ./run/main.py --yaml ruby_tok_attn_pointer_s.yml   \
--task summarization --lang_mode unilang  \
--method_name mm2seq --train_mode train_sl --dataset_type source  \
--debug 0 --multi_processing 0   \
--occupy_gpu 0-4.5       \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_tok_attn_pointer_s.yml  \
--task summarization --lang_mode unilang  \
--method_name mm2seq --train_mode test --dataset_type source  \
--debug 0 --multi_processing 0  \
--occupy_gpu 6-6        \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

# ruby_tok_attn_pointer_lr1e-3_s.yml
python -u ./run/main.py --yaml ruby_tok_attn_pointer_lr1e-3_s.yml   \
--task summarization --lang_mode unilang  \
--method_name mm2seq --train_mode train_sl --dataset_type source  \
--debug 0 --multi_processing 0   \
--occupy_gpu 0-4.5       \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_tok_attn_pointer_lr1e-3_s.yml   \
--task summarization --lang_mode unilang  \
--method_name mm2seq --train_mode test   --dataset_type source  \
--debug 0 --multi_processing 0   \
--occupy_gpu 0-4.5       \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

# ruby_tok_attn_pointer_lr1e-3_d0_s.yml
python -u ./run/main.py --yaml ruby_tok_attn_pointer_lr1e-3_d0_s.yml    \
--task summarization --lang_mode unilang  \
--method_name mm2seq --train_mode train_sl --dataset_type source  \
--debug 0 --multi_processing 0    \
--occupy_gpu 2-4.5       \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_tok_attn_pointer_lr1e-3_d0_s.yml    \
--task summarization --lang_mode unilang  \
--method_name mm2seq --train_mode test --dataset_type source  \
--debug 0 --multi_processing 0    \
--occupy_gpu 2-4.5       \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/


# ruby_tok_attn_pointer_lr1e-4_d0_s.yml
python -u ./run/main.py --yaml ruby_tok_attn_pointer_lr1e-4_d0_s.yml    \
--task summarization --lang_mode unilang  \
--method_name mm2seq --train_mode train_sl --dataset_type source  \
--debug 0 --multi_processing 0    \
--occupy_gpu 3-4.5       \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/


python -u ./run/main.py --yaml ruby_tok_attn_pointer_lr1e-4_d0_s.yml    \
--task summarization --lang_mode unilang    \
--method_name mm2seq --train_mode test --dataset_type source  \
--debug 0 --multi_processing 0      \
--occupy_gpu 3-4.5       \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

#   ruby_tok_attn_pointer_lr5e-5_d0_biF_s.yml

python -u ./run/main.py --yaml ruby_tok_attn_pointer_lr5e-5_d0_biF_s.yml   \
--task summarization --lang_mode unilang  \
--method_name mm2seq --train_mode train_sl --dataset_type source  \
--debug 0 --multi_processing 0    \
--occupy_gpu 5-4.5       \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_tok_attn_pointer_lr5e-5_d0_biF_s.yml   \
--task summarization --lang_mode unilang  \
--method_name mm2seq --train_mode test --dataset_type source  \
--debug 0 --multi_processing 0    \
--occupy_gpu 5-4.5       \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

#  ruby_tok_attn_pointer_lr1e-5_d0_biF_s.yml

python -u ./run/main.py --yaml ruby_tok_attn_pointer_lr1e-5_d0_biF_s.yml   \
--task summarization --lang_mode unilang  \
--method_name mm2seq --train_mode train_sl --dataset_type source  \
--debug 0 --multi_processing 0    \
--occupy_gpu 7-4.5       \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_tok_attn_pointer_lr1e-5_d0_biF_s.yml   \
--task summarization --lang_mode unilang  \
--method_name mm2seq --train_mode test  --dataset_type source  \
--debug 0 --multi_processing 0    \
--occupy_gpu 7-4.5       \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

# ast   attn  243g2   wanyao  deployed
python -u ./run/main.py --yaml ruby_ast_attn_s.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode train_sl --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 2-2.4   \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_ast_attn_s.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode test --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 6-6   \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

#  tok8path attn pointer  ruby
python -u ./run/main.py --yaml ruby_tok8path_attn_pointer_s.yml \
--task summarization --lang_mode unilang  \
--method_name mm2seq --train_mode train_sl --dataset_type source \
--debug 0 --multi_processing 0  \
--occupy_gpu 5-6  \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_tok8path_attn_pointer_s.yml  \
--task summarization --lang_mode unilang  \
--method_name mm2seq --train_mode train_sc --dataset_type source  \
--debug 0 --multi_processing 0  \
--occupy_gpu 5-6    \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_tok8path_attn_pointer_s.yml  \
--task summarization --lang_mode unilang  \
--method_name mm2seq --train_mode test  --dataset_type source \
--debug 0 --multi_processing 0   \
--occupy_gpu 5-6     \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_tok8path_attn_pointer_s.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode case_study  --dataset_type source \
--debug 0 --multi_processing 0  \
--occupy_gpu 7-4       \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_tok8path_attn_pointer_p0_s.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode case_study  --dataset_type source \
--debug 0 --multi_processing 0  \
--occupy_gpu 7-4   \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/



# tok8path attn pointer  ruby pr0.8  243g3
python -u ./run/main.py --yaml ruby_tok8path_attn_pointer_s_p0.8.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode train_sl --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 3-4  \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_tok8path_attn_pointer_s_p0.8.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode test --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 6-6  \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

# tok8path attn pointer  ruby pr0.6  243g6 wanyao
python -u ./run/main.py --yaml ruby_tok8path_attn_pointer_s_p0.6.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode train_sl --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 6-4  \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_tok8path_attn_pointer_s_p0.6.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode test --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 6-6   \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

# tok8path attn pointer  ruby pr0.4  243g6 wanyao
python -u ./run/main.py --yaml ruby_tok8path_attn_pointer_s_p0.4.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode train_sl --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 6-4   \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/


python -u ./run/main.py --yaml ruby_tok8path_attn_pointer_s_p0.4.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode test --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 6-6   \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

# tok8path attn pointer  ruby pr0.2  243g7
python -u ./run/main.py --yaml ruby_tok8path_attn_pointer_s_p0.2.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode train_sl --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 7-7.6  \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_tok8path_attn_pointer_s_p0.2.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode test --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 6-6  \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_tok8path_attn_pointer_s_p0.2.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode train_sc --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 5-6.2      \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_tok8path_attn_pointer_s_p0.01.yml \
--task summarization --lang_mode unilang  \
--method_name mm2seq --train_mode train_sc --dataset_type source  \
--debug 0 --multi_processing 0      \
--occupy_gpu 2-4.6        \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

# tok8path attn pointer  ruby pr0.01  243g3
python -u ./run/main.py --yaml ruby_tok8path_attn_pointer_s_p0.01.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode train_sl --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 3-4.6  \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_tok8path_attn_pointer_s_p0.01.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode test --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 6-6  \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

# tok8path  attn pointer  ruby pr0.001  243g0 wanyao
python -u ./run/main.py --yaml ruby_tok8path_attn_pointer_s_p0.001.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode train_sl --dataset_type source \
--debug 0 --multi_processing 0  \
--occupy_gpu 0-7.6     \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_tok8path_attn_pointer_s_p0.001.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode test --dataset_type source \
--debug 0 --multi_processing 0  \
--occupy_gpu 6-6     \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

# tok8path  attn pointer  ruby pr0.0001  243g0 wanyao
python -u ./run/main.py --yaml ruby_tok8path_attn_pointer_s_p0.0001.yml \
--task summarization --lang_mode unilang \
--method_name mm2seq --train_mode train_sl --dataset_type source \
--debug 0 --multi_processing 0  \
--occupy_gpu 0-9.6     \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml ruby_tok8path_attn_pointer_s_p0.0001.yml \
--task summarization --lang_mode unilang  \
--method_name mm2seq --train_mode test --dataset_type source \
--debug 0 --multi_processing 0  \
--occupy_gpu 6-6      \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

############################# ft

# ft java8javascript8php8python

src_lng=java8javascript8php8python
portion=0.0001
train_mode=test
python -u ./run/main.py --yaml ${src_lng}8ruby/tok8path-p${portion}.yml \
--task summarization --lang_mode xlang \
--method_name finetune --train_mode ${train_mode} --dataset_type all \
--debug 0 --multi_processing 0  \
--occupy_gpu 3-6 \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

src_lng=java8javascript8php8python
portion=0.001
train_mode=test
python -u ./run/main.py --yaml ${src_lng}8ruby/tok8path-p${portion}.yml \
--task summarization --lang_mode xlang \
--method_name finetune --train_mode ${train_mode} --dataset_type all \
--debug 0 --multi_processing 0    \
--occupy_gpu 3-6   \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

src_lng=java8javascript8php8python
portion=1.0
train_mode=test
python -u ./run/main.py --yaml ${src_lng}8ruby/tok8path-p${portion}.yml \
--task summarization --lang_mode xlang \
--method_name finetune --train_mode ${train_mode} --dataset_type all \
--debug 0 --multi_processing 0  \
--occupy_gpu 3-6 \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

src_lng=java8javascript8php8python
portion=0.8
train_mode=test
python -u ./run/main.py --yaml ${src_lng}8ruby/tok8path-p${portion}.yml \
--task summarization --lang_mode xlang \
--method_name finetune --train_mode ${train_mode} --dataset_type all \
--debug 0 --multi_processing 0  \
--occupy_gpu 3-6 \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

src_lng=java8javascript8php8python
portion=0
train_mode=test
python -u ./run/main.py --yaml ${src_lng}8ruby/tok8path-p${portion}.yml \
--task summarization --lang_mode xlang \
--method_name finetune --train_mode ${train_mode} --dataset_type all \
--debug 0 --multi_processing 0  \
--occupy_gpu 3-6  \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

# ft go8java8javascript8php8python pe1
# ok
src_lng=go8java8javascript8php8python
portion=0.2
train_mode=test
python -u ./run/main.py --yaml ${src_lng}8ruby_pe1/tok8path-p${portion}.yml \
--task summarization --lang_mode xlang \
--method_name finetune --train_mode ${train_mode} --dataset_type all \
--debug 0 --multi_processing 0  \
--occupy_gpu 3-6    \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

#ok
src_lng=go8java8javascript8php8python
portion=0.4
train_mode=test
python -u ./run/main.py --yaml ${src_lng}8ruby_pe1/tok8path-p${portion}.yml \
--task summarization --lang_mode xlang \
--method_name finetune --train_mode ${train_mode} --dataset_type all \
--debug 0 --multi_processing 0  \
--occupy_gpu 3-6   \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

# ok
src_lng=go8java8javascript8php8python
portion=0.6
train_mode=test
python -u ./run/main.py --yaml ${src_lng}8ruby_pe1/tok8path-p${portion}.yml \
--task summarization --lang_mode xlang \
--method_name finetune --train_mode ${train_mode} --dataset_type all \
--debug 0 --multi_processing 0  \
--occupy_gpu 3-6    \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

src_lng=go8java8javascript8php8python
portion=0.8
train_mode=test
python -u ./run/main.py --yaml ${src_lng}8ruby_pe1/tok8path-p${portion}.yml \
--task summarization --lang_mode xlang \
--method_name finetune --train_mode ${train_mode} --dataset_type all \
--debug 0 --multi_processing 0  \
--occupy_gpu 3-6      \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

src_lng=go8java8javascript8php8python
portion=0
train_mode=test
python -u ./run/main.py --yaml ${src_lng}8ruby_pe1/tok8path-p${portion}.yml \
--task summarization --lang_mode xlang \
--method_name finetune --train_mode ${train_mode} --dataset_type all \
--debug 0 --multi_processing 0  \
--occupy_gpu 3-6     \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

# ft src python 5g
src_lng=python
portion=1.0
train_mode=None
python -u ./run/main.py --yaml ${src_lng}8ruby/tok8path-p${portion}.yml \
--task summarization --lang_mode xlang \
--method_name finetune --train_mode ${train_mode} --dataset_type all \
--debug 0 --multi_processing 0 \
--occupy_gpu 3-5 \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

src_lng=python
portion=0
train_mode=test
python -u ./run/main.py --yaml ${src_lng}8ruby/tok8path-p${portion}.yml \
--task summarization --lang_mode xlang \
--method_name finetune --train_mode ${train_mode} --dataset_type all \
--debug 0 --multi_processing 0  \
--occupy_gpu 6-6 \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

src_lng=python
portion=0
train_mode=test
python -u ./run/main.py --yaml ${src_lng}8ruby_pe1/tok8path-p${portion}.yml \
--task summarization --lang_mode xlang \
--method_name finetune --train_mode ${train_mode} --dataset_type all \
--debug 0 --multi_processing 0  \
--occupy_gpu 5-7  \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

src_lng=python
portion=0
train_mode=case_study
python -u ./run/main.py --yaml ${src_lng}8ruby_pe1/tok8path-p${portion}.yml \
--task summarization --lang_mode xlang \
--method_name finetune --train_mode ${train_mode} --dataset_type all \
--debug 0 --multi_processing 0   \
--occupy_gpu 7-4     \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

# ft src php 10g
src_lng=php
portion=1.0
train_mode=None
python -u ./run/main.py --yaml ${src_lng}8ruby/tok8path-p${portion}.yml \
--task summarization --lang_mode xlang \
--method_name finetune --train_mode ${train_mode} --dataset_type all \
--debug 0 --multi_processing 0 \
--occupy_gpu 5-10  \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

src_lng=php
portion=0
train_mode=test
python -u ./run/main.py --yaml ${src_lng}8ruby/tok8path-p${portion}.yml \
--task summarization --lang_mode xlang \
--method_name finetune --train_mode ${train_mode} --dataset_type all \
--debug 0 --multi_processing 0 \
--occupy_gpu 6-6  \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

src_lng=php
portion=0
train_mode=test
python -u ./run/main.py --yaml ${src_lng}8ruby_pe1/tok8path-p${portion}.yml  \
--task summarization --lang_mode xlang  \
--method_name finetune --train_mode ${train_mode} --dataset_type all \
--debug 0 --multi_processing 0  \
--occupy_gpu 5-7   \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

src_lng=php
portion=0
train_mode=case_study
python -u ./run/main.py --yaml ${src_lng}8ruby_pe1/tok8path-p${portion}.yml  \
--task summarization --lang_mode xlang  \
--method_name finetune --train_mode ${train_mode} --dataset_type all \
--debug 0 --multi_processing 0  \
--occupy_gpu 7-4     \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

# ft src javascript   7.4g
src_lng=javascript
portion=1.0
train_mode=None
python -u ./run/main.py --yaml ${src_lng}8ruby/tok8path-p${portion}.yml \
--task summarization --lang_mode xlang \
--method_name finetune --train_mode ${train_mode} --dataset_type all \
--debug 0 --multi_processing 0 \
--occupy_gpu 7-7  \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

src_lng=javascript
portion=0
train_mode=test
python -u ./run/main.py --yaml ${src_lng}8ruby/tok8path-p${portion}.yml \
--task summarization --lang_mode xlang \
--method_name finetune --train_mode ${train_mode} --dataset_type all \
--debug 0 --multi_processing 0 \
--occupy_gpu 6-6  \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

src_lng=javascript
portion=0
train_mode=test
python -u ./run/main.py --yaml ${src_lng}8ruby_pe1/tok8path-p${portion}.yml \
--task summarization --lang_mode xlang \
--method_name finetune --train_mode ${train_mode} --dataset_type all \
--debug 0 --multi_processing 0 \
--occupy_gpu 5-7    \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

# ft src java   5.4 g  240g1
src_lng=java
portion=1.0
train_mode=None
python -u ./run/main.py --yaml ${src_lng}8ruby/tok8path-p${portion}.yml \
--task summarization --lang_mode xlang \
--method_name finetune --train_mode ${train_mode} --dataset_type all \
--debug 0 --multi_processing 0 \
--occupy_gpu 1-5  \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/


src_lng=java
portion=0
train_mode=test
python -u ./run/main.py --yaml ${src_lng}8ruby/tok8path-p${portion}.yml \
--task summarization --lang_mode xlang \
--method_name finetune --train_mode ${train_mode} --dataset_type all \
--debug 0 --multi_processing 0 \
--occupy_gpu 6-6  \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

src_lng=java
portion=0
train_mode=test
python -u ./run/main.py --yaml ${src_lng}8ruby_pe1/tok8path-p${portion}.yml \
--task summarization --lang_mode xlang  \
--method_name finetune --train_mode ${train_mode} --dataset_type all \
--debug 0 --multi_processing 0 \
--occupy_gpu 5-7   \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

src_lng=java
portion=0
train_mode=case_study
python -u ./run/main.py --yaml ${src_lng}8ruby_pe1/tok8path-p${portion}.yml \
--task summarization --lang_mode xlang  \
--method_name finetune --train_mode ${train_mode} --dataset_type all \
--debug 0  --multi_processing 0  \
--occupy_gpu  7-4   \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

####
src_lng=javascript
portion=1.0
train_mode=test
python -u ./run/main.py --yaml ${src_lng}8ruby_pe1/tok8path-p${portion}.yml \
--task summarization --lang_mode xlang \
--method_name finetune --train_mode ${train_mode} --dataset_type all \
--debug 0 --multi_processing 0 \
--occupy_gpu 5-7  \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

src_lng=javascript
portion=0
train_mode=case_study
python -u ./run/main.py --yaml ${src_lng}8ruby_pe1/tok8path-p${portion}.yml \
--task summarization --lang_mode xlang \
--method_name finetune --train_mode ${train_mode} --dataset_type all \
--debug 0 --multi_processing 0 \
--occupy_gpu 7-4  \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

# ft src go   6.8 g    243g2 wanyao
src_lng=go
portion=1.0
train_mode=None
python -u ./run/main.py --yaml ${src_lng}8ruby/tok8path-p${portion}.yml \
--task summarization --lang_mode xlang \
--method_name finetune --train_mode ${train_mode} --dataset_type all \
--debug 0 --multi_processing 0 \
--occupy_gpu 2-6.4  \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

src_lng=go
portion=0
train_mode=test
python -u ./run/main.py --yaml ${src_lng}8ruby/tok8path-p${portion}.yml \
--task summarization --lang_mode xlang \
--method_name finetune --train_mode ${train_mode} --dataset_type all \
--debug 0 --multi_processing 0 \
--occupy_gpu 7-6.4   \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

src_lng=go
portion=1.0
train_mode=None
python -u ./run/main.py --yaml ${src_lng}8ruby_pe1/tok8path-p${portion}.yml   \
--task summarization --lang_mode xlang     \
--method_name finetune --train_mode ${train_mode} --dataset_type all      \
--debug 0 --multi_processing 0  \
--occupy_gpu 7-6.5    \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

src_lng=go
portion=0
train_mode=case_study
python -u ./run/main.py --yaml ${src_lng}8ruby_pe1/tok8path-p${portion}.yml   \
--task summarization --lang_mode xlang     \
--method_name finetune --train_mode ${train_mode} --dataset_type all      \
--debug 0 --multi_processing 0  \
--occupy_gpu 7-4       \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

src_lng=go
portion=0
train_mode=test
python -u ./run/main.py --yaml ${src_lng}8ruby_pe1/tok8path-p${portion}.yml \
--task summarization --lang_mode xlang \
--method_name finetune --train_mode ${train_mode} --dataset_type all \
--debug 0 --multi_processing 0 \
--occupy_gpu 5-6.4   \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

#############################      kd
# kd teacher_python  243g0 wanyao

python -u ./run/main.py --yaml source_java_javascript_php_python/teacher_python.yml \
--task summarization --lang_mode xlang \
--method_name kd_mm2seq --train_mode train_sl --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 0-7.6    \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/


# kd teacher_php 243g2 wanyao

python -u ./run/main.py --yaml source_java_javascript_php_python/teacher_php.yml \
--task summarization --lang_mode xlang \
--method_name kd_mm2seq --train_mode train_sl --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 2-10  \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

# kd teacher_javascript 243g2 wanyao
python -u ./run/main.py --yaml source_java_javascript_php_python/teacher_javascript.yml \
--task summarization --lang_mode xlang \
--method_name kd_mm2seq --train_mode train_sl --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 2-7  \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/



# kd teacher_java 243g0 wanyao
python -u ./run/main.py --yaml source_java_javascript_php_python/teacher_java.yml \
--task summarization --lang_mode xlang \
--method_name kd_mm2seq --train_mode train_sl --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 0-5  \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

# kd student 243g2 wanyao
python -u ./run/main.py --yaml source_java_javascript_php_python/student.yml \
--task summarization --lang_mode xlang \
--method_name kd_mm2seq --train_mode train_sl --dataset_type source  \
--debug 0 --multi_processing 0 \
--occupy_gpu 2-10   \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

# kd train_sl_ft ruby pr1.0  243g0 wanyao
python -u ./run/main.py --yaml source_java_javascript_php_python/ft_ruby_with_student_p1.0.yml \
--task summarization --lang_mode xlang \
--method_name kd_mm2seq --train_mode train_sl_ft --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 0-7.6  \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml source_java_javascript_php_python/ft_ruby_with_student_p1.0.yml \
--task summarization --lang_mode xlang \
--method_name kd_mm2seq --train_mode test --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 6-6  \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

# kd train_sl_ft ruby pr0.8  243g2 wanyao
python -u ./run/main.py --yaml source_java_javascript_php_python/ft_ruby_with_student_p0.8.yml \
--task summarization --lang_mode xlang \
--method_name kd_mm2seq --train_mode train_sl_ft --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 2-7.6  \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml source_java_javascript_php_python/ft_ruby_with_student_p0.8.yml \
--task summarization --lang_mode xlang \
--method_name kd_mm2seq --train_mode test --dataset_type source \
--debug 0 --multi_processing 0   \
--occupy_gpu 6-6  \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

# kd train_sl_ft ruby pr0.6  240g3 wanyao
python -u ./run/main.py --yaml source_java_javascript_php_python/ft_ruby_with_student_p0.6.yml \
--task summarization --lang_mode xlang \
--method_name kd_mm2seq --train_mode train_sl_ft --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 3-7.6  \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml source_java_javascript_php_python/ft_ruby_with_student_p0.6.yml \
--task summarization --lang_mode xlang \
--method_name kd_mm2seq --train_mode test --dataset_type source \
--debug 0 --multi_processing 0   \
--occupy_gpu 6-6  \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

# kd train_sl_ft ruby pr0.4  243g0 wanyao
python -u ./run/main.py --yaml source_java_javascript_php_python/ft_ruby_with_student_p0.4.yml \
--task summarization --lang_mode xlang \
--method_name kd_mm2seq --train_mode train_sl_ft --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 0-7.6  \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml source_java_javascript_php_python/ft_ruby_with_student_p0.4.yml \
--task summarization --lang_mode xlang \
--method_name kd_mm2seq --train_mode test  --dataset_type source \
--debug 0 --multi_processing 0 \
--occupy_gpu 6-6  \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

# kd train_sl_ft ruby pr0.2  243g2 wanyao
python -u ./run/main.py --yaml source_java_javascript_php_python/ft_ruby_with_student_p0.2.yml \
--task summarization --lang_mode xlang \
--method_name kd_mm2seq --train_mode train_sl_ft --dataset_type source \
--debug 0 --multi_processing 0  \
--occupy_gpu 2-7.6  \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml source_java_javascript_php_python/ft_ruby_with_student_p0.2.yml \
--task summarization --lang_mode xlang \
--method_name kd_mm2seq --train_mode test --dataset_type source \
--debug 0 --multi_processing 0  \
--occupy_gpu 6-6  \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

# kd train_sl_ft ruby pr0.01  240g3 wanyao
python -u ./run/main.py --yaml source_java_javascript_php_python/ft_ruby_with_student_p0.01.yml \
--task summarization --lang_mode xlang \
--method_name kd_mm2seq --train_mode train_sl_ft --dataset_type source \
--debug 0 --multi_processing 0  \
--occupy_gpu 3-7.6    \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml source_java_javascript_php_python/ft_ruby_with_student_p0.01.yml \
--task summarization --lang_mode xlang \
--method_name kd_mm2seq --train_mode test --dataset_type source \
--debug 0 --multi_processing 0  \
--occupy_gpu 6-6    \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

# kd train_sl_ft ruby pr0.001  240g3 wanyao
python -u ./run/main.py --yaml source_java_javascript_php_python/ft_ruby_with_student_p0.001.yml \
--task summarization --lang_mode xlang \
--method_name kd_mm2seq --train_mode train_sl_ft --dataset_type source \
--debug 0 --multi_processing 0  \
--occupy_gpu 3-7.6    \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml source_java_javascript_php_python/ft_ruby_with_student_p0.001.yml \
--task summarization --lang_mode xlang \
--method_name kd_mm2seq --train_mode test --dataset_type source \
--debug 0 --multi_processing 0  \
--occupy_gpu 6-6    \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

# kd train_sl_ft ruby pr0.0001  240g3 wanyao
python -u ./run/main.py --yaml source_java_javascript_php_python/ft_ruby_with_student_p0.0001.yml \
--task summarization --lang_mode xlang \
--method_name kd_mm2seq --train_mode train_sl_ft --dataset_type source \
--debug 0 --multi_processing 0  \
--occupy_gpu 3-7.6    \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

python -u ./run/main.py --yaml source_java_javascript_php_python/ft_ruby_with_student_p0.0001.yml \
--task summarization --lang_mode xlang \
--method_name kd_mm2seq --train_mode test --dataset_type source \
--debug 0 --multi_processing 0  \
--occupy_gpu 6-6    \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/


################# dtrl

# test php p1.0
python -u ./run/main.py --yaml php8ruby/tok8path-p1.0.yml \
--task summarization --lang_mode xlang \
--method_name dtrl --train_mode test --dataset_type target \
--debug 0 --multi_processing 0  \
--occupy_gpu 2-10    \
--log_root_dir /data/sjd/d/p_d/fse20all/100_small/

