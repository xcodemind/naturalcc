
nohup python -u ./run/retrieval/unilang/deepcs/main.py --yaml python.yml --task retrieval --lang_mode unilang --method_name deepcs --train_mode train_sl --dataset_type source >./run/retrieval/unilang/deepcs/retrieval_python.log
nohup python -u ./run/retrieval/unilang/deepcs/main.py --yaml java.yml --task retrieval --lang_mode unilang --method_name deepcs --train_mode train_sl --dataset_type source >./run/retrieval/unilang/deepcs/retrieval_java.log
nohup python -u ./run/retrieval/unilang/deepcs/main.py --yaml javascript.yml --task retrieval --lang_mode unilang --method_name deepcs --train_mode train_sl --dataset_type source >./run/retrieval/unilang/deepcs/retrieval_javascript.log
nohup python -u ./run/retrieval/unilang/deepcs/main.py --yaml php.yml --task retrieval --lang_mode unilang --method_name deepcs --train_mode train_sl --dataset_type source >./run/retrieval/unilang/deepcs/retrieval_php.log
nohup python -u ./run/retrieval/unilang/deepcs/main.py --yaml go.yml --task retrieval --lang_mode unilang --method_name deepcs --train_mode train_sl --dataset_type source >./run/retrieval/unilang/deepcs/retrieval_go.log
nohup python -u ./run/retrieval/unilang/deepcs/main.py --yaml ruby.yml --task retrieval --lang_mode unilang --method_name deepcs --train_mode train_sl --dataset_type source >./run/retrieval/unilang/deepcs/retrieval_ruby.log

