
nohup python -u ./run/retrieval/unilang/cnn/main.py --yaml python.yml --task retrieval --lang_mode unilang --method_name cnn --train_mode train_sl --dataset_type source >./run/retrieval/unilang/cnn/retrieval_python.log
nohup python -u ./run/retrieval/unilang/cnn/main.py --yaml java.yml --task retrieval --lang_mode unilang --method_name cnn --train_mode train_sl --dataset_type source >./run/retrieval/unilang/cnn/retrieval_java.log
nohup python -u ./run/retrieval/unilang/cnn/main.py --yaml javascript.yml --task retrieval --lang_mode unilang --method_name cnn --train_mode train_sl --dataset_type source >./run/retrieval/unilang/cnn/retrieval_javascript.log
nohup python -u ./run/retrieval/unilang/cnn/main.py --yaml php.yml --task retrieval --lang_mode unilang --method_name cnn --train_mode train_sl --dataset_type source >./run/retrieval/unilang/cnn/retrieval_php.log
nohup python -u ./run/retrieval/unilang/cnn/main.py --yaml go.yml --task retrieval --lang_mode unilang --method_name cnn --train_mode train_sl --dataset_type source >./run/retrieval/unilang/cnn/retrieval_go.log
nohup python -u ./run/retrieval/unilang/cnn/main.py --yaml ruby.yml --task retrieval --lang_mode unilang --method_name cnn --train_mode train_sl --dataset_type source >./run/retrieval/unilang/cnn/retrieval_ruby.log

