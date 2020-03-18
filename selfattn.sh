
nohup python -u ./run/retrieval/unilang/selfattn/main.py --yaml python.yml --task retrieval --lang_mode unilang --method_name selfattn --train_mode train_sl --dataset_type source >./run/retrieval/unilang/selfattn/retrieval_python.log
nohup python -u ./run/retrieval/unilang/selfattn/main.py --yaml java.yml --task retrieval --lang_mode unilang --method_name selfattn --train_mode train_sl --dataset_type source >./run/retrieval/unilang/selfattn/retrieval_java.log
nohup python -u ./run/retrieval/unilang/selfattn/main.py --yaml javascript.yml --task retrieval --lang_mode unilang --method_name selfattn --train_mode train_sl --dataset_type source >./run/retrieval/unilang/selfattn/retrieval_javascript.log
nohup python -u ./run/retrieval/unilang/selfattn/main.py --yaml php.yml --task retrieval --lang_mode unilang --method_name selfattn --train_mode train_sl --dataset_type source >./run/retrieval/unilang/selfattn/retrieval_php.log
nohup python -u ./run/retrieval/unilang/selfattn/main.py --yaml go.yml --task retrieval --lang_mode unilang --method_name selfattn --train_mode train_sl --dataset_type source >./run/retrieval/unilang/selfattn/retrieval_go.log
nohup python -u ./run/retrieval/unilang/selfattn/main.py --yaml ruby.yml --task retrieval --lang_mode unilang --method_name selfattn --train_mode train_sl --dataset_type source >./run/retrieval/unilang/selfattn/retrieval_ruby.log

