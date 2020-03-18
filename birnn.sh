nohup python -u ./run/retrieval/unilang/birnn/main.py --yaml python.yml --task retrieval --lang_mode unilang --method_name birnn --train_mode train_sl --dataset_type source >./run/retrieval/unilang/birnn/retrieval_python.log
nohup python -u ./run/retrieval/unilang/birnn/main.py --yaml java.yml --task retrieval --lang_mode unilang --method_name birnn --train_mode train_sl --dataset_type source >./run/retrieval/unilang/birnn/retrieval_java.log

nohup python -u ./run/retrieval/unilang/birnn/main.py --yaml javascript.yml --task retrieval --lang_mode unilang --method_name birnn --train_mode train_sl --dataset_type source >./run/retrieval/unilang/birnn/retrieval_javascript.log
nohup python -u ./run/retrieval/unilang/birnn/main.py --yaml php.yml --task retrieval --lang_mode unilang --method_name birnn --train_mode train_sl --dataset_type source >./run/retrieval/unilang/birnn/retrieval_php.log
nohup python -u ./run/retrieval/unilang/birnn/main.py --yaml go.yml --task retrieval --lang_mode unilang --method_name birnn --train_mode train_sl --dataset_type source >./run/retrieval/unilang/birnn/retrieval_go.log
nohup python -u ./run/retrieval/unilang/birnn/main.py --yaml ruby.yml --task retrieval --lang_mode unilang --method_name birnn --train_mode train_sl --dataset_type source >./run/retrieval/unilang/birnn/retrieval_ruby.log

