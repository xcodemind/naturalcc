如果使用小写形式，运行prepare_data_for_moses_lowercase.py,(如果不强制使用小写形式，运行prepare_data_for_moses.py, )
再在run_script.sh中全文替换prefix为prepare_data_for_moses_lowercase.py中用到的prefix
再参照 run_script.sh，最后运行dump_moses_pred.py,
将得到的文件路径输入当前目录下的evaluate.py 得到metric



