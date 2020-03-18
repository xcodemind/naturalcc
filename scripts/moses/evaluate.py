from tabulate import tabulate
from eval.summarization.bleu.bleu import Bleu
from eval.summarization.cider.cider import Cider
from eval.summarization.meteor.meteor import Meteor
from eval.summarization.rouge.rouge import Rouge


def get_metric(metrics,trgs,preds):
    # eval score
    if 'bleu' in metrics:
        _, bleu = Bleu(4).compute_score(trgs, preds)
        bleu1, bleu2, bleu3, bleu4, = bleu
        # print('bleu1-: ', bleu1)
    else:
        bleu1, bleu2, bleu3, bleu4 = \
            [0.0] * len(trgs), [0.0] * len(trgs), [0.0] * len(trgs), [0.0] * len(trgs)
    if 'meteor' in metrics:
        _, meteor = Meteor().compute_score(trgs, preds)
    else:
        meteor = [0.0] * len(trgs)
    if 'rouge' in metrics:
        # print('rouge-trgs: ', trgs)
        # print('rouge-preds: ', preds)
        rouge, _ = Rouge().compute_score(trgs, preds)  #
        # assert False
        # print('rouge: ', rouge)
        # print('_: ', _)
        rouge1, rouge2, rouge3, rouge4, rougel, _, _, _ = [[i] for i in rouge]
        # print('rouge1-: ', rouge1)
    else:
        rouge1, rouge2, rouge3, rouge4, rougel = [0.0] * len(trgs), [0.0] * len(trgs), \
                                                 [0.0] * len(trgs), [0.0] * len(trgs), \
                                                 [0.0] * len(trgs),
    if 'cider' in metrics:
        print("cal_cider")
        _, cider = Cider().compute_score(trgs, preds)
    else:
        cider = [0.0] * len(trgs)


    bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider, = \
        map(lambda array: sum(array) / len(array), (bleu1, bleu2, bleu3, bleu4, meteor,
                                                    rouge1, rouge2, rouge3, rouge4, rougel, cider,))

    return bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider

if __name__ == "__main__":
    # python -m  scripts.moses.evaluate

    # result_file = '/data/wanyao/work/baseline/workinglow/moses_data_lower.fse20ruby100_test.translated.com.pred'
    # result_file = '/data/wanyao/work/baseline/workinglow/moses_data_lower.fse20ruby100_v2_test.translated.com.pred'
    # result_file = '/data/wanyao/work/baseline/workinglow/moses_data_lower.fse20ruby100_v2_test.translated.com_target_code.pred'
    # result_file = '/data/wanyao/work/baseline/workinglow/moses_data_lower.fse20ruby100_v3_test.translated.com_target_com.pred'
    # result_file = '/data/wanyao/work/baseline/workinglow/moses_data_lower.fse20ruby200_test.translated.com_target_com.pred'
    result_file = '/data/wanyao/work/baseline/workinglow/moses_data_lower.fse20ruby100small_p0.01_test.translated.com_target_com.pred'


    with open(result_file, 'r', encoding="utf-8") as file:
        lines = file.readlines()
        res, gts = {}, {}
        for i, line in enumerate(lines):
            if i%3 == 1:
                res[int(line.strip('\n').split('=====')[0])] = [line.strip('\n').split('=====')[2].strip()]
            elif i%3 == 2:
                gts[int(line.strip('\n').split('=====')[0])] = [line.strip('\n').split('=====')[2].strip()] # 不能用冒号分割，因为有可能：是comment的一部分
    print('res[0]: ', res[0])
    print('gts[0]: ', gts[0])
    print('res[1]: ', res[1])
    print('gts[1]: ', gts[1])
    print('res[2]: ', res[2])
    print('gts[2]: ', gts[2])

    print('len(res): ', len(res))
    print('len(gts): ', len(gts))

    bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider =\
        get_metric(['bleu', 'meteor', 'rouge', 'cider'],gts,res)

    headers = ['B1', 'B2', 'B3', 'B4', 'Meteor', 'R1', 'R2', 'R3', 'R4', 'RL', 'Cider']
    result_table = [[round(i, 4) for i in [bleu1, bleu2, bleu3, bleu4, meteor,
                                           rouge1, rouge2, rouge3, rouge4, rougel, cider]]]
    print('Evaluation results:\n{}'.format(tabulate(result_table, headers=headers,
                                                          tablefmt='github')))

    print("\n\n")
    print(headers)
    print([round(i, 4) for i in [bleu1, bleu2, bleu3, bleu4, meteor,
                                           rouge1, rouge2, rouge3, rouge4, rougel, cider]])

    print("result_file: ",result_file)