
# target_file = "/data/wanyao/work/baseline/corpus/githubpythonformoseslow.test_ct.true.com"
# moses_pred_file = "/data/wanyao/work/baseline/workinglow/githubpythonformoseslow.test_ct.translated.com"
# output_preds_file = "/data/wanyao/work/ghproj_d/code-sum-mm/github-python/baseline/moseslow.pred"

# target_file = "/data/wanyao/work/baseline/corpus/githubpythonformoses.test_ct.true.com"
# moses_pred_file = "/data/wanyao/work/baseline/working/githubpythonformoses.test_ct.translated.com"
# output_preds_file = "/data/wanyao/work/ds/codedata/debug-data-path/sixshouldok/baselineresult/moses.pred"

# target_file = "/data/wanyao/work/baseline/corpus/githubpythonformoseslow.c_aa0_test_ct.true.code"
# moses_pred_file = "/data/wanyao/work/baseline/workinglow/githubpythonformoseslow.c_aa0_test_ct.translated.com"
# output_preds_file = "/data/wanyao/work/ghproj_d/code-sum-mm/github-python/baseline/moseslow_c_aa0.pred"

# target_file = "/data/wanyao/work/baseline/corpus/githubpythonformoseslow.csharp5test_ct.true.code"
# moses_pred_file = "/data/wanyao/work/baseline/workinglow/githubpythonformoseslow.csharp5test_ct.translated.com"
# output_preds_file = "/data/wanyao/work/ghproj_d/code-sum-mm/github-python/baseline/moseslow_csharp5.pred"

# target_file = "/data/wanyao/work/baseline/corpus/moses_data_lower.fse20ruby100_test.true.com"
# moses_pred_file = "/data/wanyao/work/baseline/workinglow/moses_data_lower.fse20ruby100_test.translated.com"
# output_preds_file = moses_pred_file + '.pred'

# target_file = "/data/wanyao/work/baseline/corpus/moses_data_lower.fse20ruby100_v2_test.true.com"
# moses_pred_file = "/data/wanyao/work/baseline/workinglow/moses_data_lower.fse20ruby100_v2_test.translated.com"
# output_preds_file = moses_pred_file + '.pred'

# target_file = "/data/wanyao/work/baseline/corpus/moses_data_lower.fse20ruby100_v2_test.true.code" # 本来应该和com比较，现在这里和code比较试试
# moses_pred_file = "/data/wanyao/work/baseline/workinglow/moses_data_lower.fse20ruby100_v2_test.translated.com"
# output_preds_file = moses_pred_file + '_target_'+target_file.split('.')[-1]+'.pred'

# target_file = "/data/wanyao/work/baseline/corpus/moses_data_lower.fse20ruby100_v3_test.true.com"
# moses_pred_file = "/data/wanyao/work/baseline/workinglow/moses_data_lower.fse20ruby100_v3_test.translated.com"
# output_preds_file = moses_pred_file + '_target_'+target_file.split('.')[-1]+'.pred'

# target_file = "/data/wanyao/work/baseline/corpus/moses_data_lower.fse20ruby200_test.true.com"
# moses_pred_file = "/data/wanyao/work/baseline/workinglow/moses_data_lower.fse20ruby200_test.translated.com"
# output_preds_file = moses_pred_file + '_target_'+target_file.split('.')[-1]+'.pred'

target_file = "/data/wanyao/work/baseline/corpus/moses_data_lower.fse20ruby100small_p0.01_test.true.com"
moses_pred_file = "/data/wanyao/work/baseline/workinglow/moses_data_lower.fse20ruby100small_p0.01_test.translated.com"
output_preds_file = moses_pred_file + '_target_'+target_file.split('.')[-1]+'.pred'

def dump_preds(  moses_lines, target_lines, output_preds_file):
    print("enter_dump_preds")
    assert len(moses_lines) == len(target_lines)
    src = "src"
    print("open :\n",output_preds_file)
    with open(output_preds_file, "w") as f:
        for i in range(len(moses_lines)):
            print("{}/{}".format(i,len(moses_lines)))
            pred = moses_lines[i].strip()
            # if pred[-1] != "\n":
            #     assert False
            # else:
            #     print("pred end enter")
            tgt = target_lines[i].strip()

            # f.write(str(i) + ": src:  "+ " ".join(src) + '\n')
            # f.write(str(i) + ": pre: " + " ".join(pred) + '\n')
            # f.write(str(i) + ": tgt:  "+ " ".join(tgt) + '\n')

            # f.write(str(i) + ": src: "+ src + '\n')
            # f.write(str(i) + ": pre: " + pred + '\n')
            # f.write(str(i) + ": tgt: "+ tgt + '\n')

            # f.write(str(i) + '=====src=====' + ' '.join(src) + '\n')
            # f.write(str(i) + '=====pre=====' + ' '.join(pred) + '\n')
            # f.write(str(i) + '=====tgt=====' + ' '.join(tgt) + '\n')
            f.write(str(i) + '=====src=====' +src + '\n')
            f.write(str(i) + '=====pre=====' + pred + '\n')
            f.write(str(i) + '=====tgt=====' + tgt + '\n')


def main(target_file,moses_pred_file,output_preds_file):
    with open(moses_pred_file,"r",encoding="utf-8") as f_moses:
        moses_lines = f_moses.readlines()
    with open(target_file,"r",encoding="utf-8") as f_target:
        target_lines = f_target.readlines()
        print("len(moses_lines): ",len(moses_lines))
        print("len(target_lines): ",len(target_lines))
    dump_preds(  moses_lines, target_lines, output_preds_file)

# python -m  scripts.moses.dump_moses_pred
main(target_file,moses_pred_file,output_preds_file)
print("output_preds_file: ",output_preds_file)