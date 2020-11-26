import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import os
import copy
# from brokenaxes import brokenaxes # 坐标轴截断
# from ncc.utils.util_vis import  plot_2subfig_bleu_meteor_rougeL_cider
# from  ncc.utils.util  import load_json
import json
from matplotlib import gridspec

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.switch_backend('agg')

def plot_2subfig_bleu_meteor_rougeL_cider(x,x_label,y_label_list,font_size,bleu1_list,meteor_list,rouge_list,
                                        cider_list,str_tick_list,save_path,color_dict,tag=None,b_loc=None,c_loc=None,
                                          line_icon=None):
    fig = plt.figure()


    ymajorFormatter = FormatStrFormatter('%1.1f')  # 设置y轴标签文本的格式

    gs = gridspec.GridSpec(2,1, height_ratios=[1,2])

    print("=====\nx:\n",x)
    print("bleu1_list:\n",bleu1_list)
    print("meteor_list:\n",meteor_list)
    print("rouge_list:\n",rouge_list)
    print("cider_list:\n",cider_list)

    # ax = plt.subplot(211)
    ax = plt.subplot(gs[0])

    if y_label_list[0] is not None:
        ax.set_ylabel(y_label_list[0], fontsize=font_size)




    pc = ax.plot(x, cider_list, marker=line_icon['cider'], linewidth=2.0, markersize=15, label='C', color=color_dict["cider"])
    ax.tick_params(labelsize=font_size-1)
    ax.set_xticks(x)
    ax.set_xticklabels( str_tick_list)

    if tag == "percent":
        ax.set_ylim([min(cider_list)-0.5,max(cider_list)+0.5 ])  # 设置y轴刻度的范围
        fig.legend([pc[0]], ["C"], fontsize=font_size - 5, ncol=4,
                   loc=(0.825, 0.885))  # , loc='upper center'
    else:
        # ax.set_ylim([-0.4  ,5])  # 设置y轴刻度的范围
        # ax.set_ylim([0  ,0.35 ])  # 设置y轴刻度的范围
        ax.set_ylim([min(cider_list) - 0.1, max(cider_list) + 0.1])  # 设置y轴刻度的范围
        # fig.legend([pc[0]], ["C"], fontsize=font_size - 5, ncol=4,
        #            loc=(0.215, 0.645 ))  # , loc='upper center'
        # fig.legend([pc[0]], ["CIDER"], fontsize=font_size - 5, ncol=4,
        #            loc=(0.747, 0.878 ))  # , loc='upper center'
        fig.legend([pc[0]], ["CIDER"], fontsize=font_size - 5, ncol=4,
                   loc=c_loc )  # , loc='upper center'
    # fig.legend([pc[0]], ["C"], fontsize=font_size - 5, ncol=4,
    #            loc=(0.825, 0.885))  # , loc='upper center'
    # fig.legend([pc[0]], ["C"], fontsize=font_size - 5, ncol=4,
    #            loc= 3   )  # , loc='upper center'


    ax.yaxis.set_major_formatter(ymajorFormatter)
    ax.grid()

    # ax = plt.subplot(212)
    ax = plt.subplot(gs[1])


    if y_label_list[1] is not None:
        ax.set_ylabel(y_label_list[1], fontsize=font_size)
    pb = ax.plot(x, bleu1_list, marker=line_icon['bleu1'], linewidth=2.0, markersize=15, label='B-1',color=color_dict["bleu1"])
    pm = ax.plot(x, meteor_list, marker=line_icon['meteor'], linewidth=2.0, markersize=15, label='M',color=color_dict["meteor"]) # "2"
    pr = ax.plot(x, rouge_list, marker=line_icon['rouge'], linewidth=2.0, markersize=15, label='R',color=color_dict["rouge"])
    ax.tick_params(labelsize=font_size-1)
    ax.set_xticks(x)
    ax.set_xticklabels( str_tick_list)

    if tag == "percent":
        ax.set_ylim( [min(min(bleu1_list),min(meteor_list),min(rouge_list))-0.05,
                      max(max(bleu1_list),max(meteor_list),max(rouge_list))+0.05  ]  )  # 设置y轴刻度的范围
    else:
        # ax.set_ylim( [-0.1   , 1 ]  )  # 设置y轴刻度的范围
        # ax.set_ylim( [0   , 0.6   ]  )  # 设置y轴刻度的范围
        # ax.set_ylim( [0   , 0.2  ]  )  # 设置y轴刻度的范围
        # ax.set_ylim( [min(min(bleu1_list),min(meteor_list),min(rouge_list))-0.05,
        #               max(max(bleu1_list),max(meteor_list),max(rouge_list))+0.08  ]  )  # 设置y轴刻度的范围
        ax.set_ylim( [min(min(bleu1_list),min(meteor_list),min(rouge_list))-0.05,
                      max(max(bleu1_list),max(meteor_list),max(rouge_list))+0.2  ]  )  # 设置y轴刻度的范围

    ax.set_xlabel(x_label, fontsize=font_size)
    # fig.legend([pb[0],pm[0],pr[0],pc[0]],["B-1","M","R","C" ],fontsize=font_size-5,ncol=4,loc=(0.2,0.88   )) #, loc='upper center'
    if tag == "percent":
        fig.legend([pb[0],pm[0],pr[0]],["B-1","M","R" ],
                   fontsize=font_size-5,ncol=4,loc=(0.43,0.335   )) #, loc='upper center'
    else:
        # fig.legend([pb[0],pm[0],pr[0]],["B-1","M","R" ],
        #            fontsize=font_size-5,ncol=4,loc=(0.43,0.435   )) #, loc='upper center'

        # fig.legend([pb[0],pm[0],pr[0]],["B-1","M","R" ],
        #            fontsize=font_size-5,ncol=4,loc=(0.43,0.425   )) #, loc='upper center'
        # fig.legend([pb[0],pm[0],pr[0]],["B-1","M","R" ],
        #            fontsize=font_size-5,ncol=4,loc= (0.22,0.20  )) #, loc='upper center'
        # fig.legend([pb[0],pm[0],pr[0]],["B-1","M","R" ],
        #            fontsize=font_size-5,ncol=4,loc=  (0.44,0.44 ) ) #, loc=  'upper right'
        # fig.legend([pb[0],pm[0],pr[0]],["BLEU-1","METEOR","ROUGE" ],
        #            fontsize=font_size-5,ncol=3,columnspacing=0.03 ,loc=  (0.22,0.44 ) ) #, loc=  'upper right'

        # fig.legend([pb[0],pm[0],pr[0]],["BLEU-1","METEOR","ROUGE" ],
        #            fontsize=font_size-5,ncol=1,columnspacing=0.04,labelspacing=0.4,
        #            loc=  (0.71 ,0.42 ) ) #, loc=  'upper right'
        fig.legend([pb[0],pm[0],pr[0]],["BLEU-1","METEOR","ROUGE" ],
                   fontsize=font_size-5,ncol=1,columnspacing=0.04,labelspacing=0.4,
                   loc=  b_loc  ) #, loc=  'upper right'

    ax.yaxis.set_major_formatter(ymajorFormatter)
    ax.grid()

    plt.tight_layout(w_pad=-5.1, rect=[0, 0, 1, 1])
    fig.savefig(save_path)


def get_retr_metric( d):
    bleu1 = np.mean( np.array(d["bleu1"])  )
    meteor = np.mean( np.array(d["meteor"])  )
    rougeL = np.mean( np.array(d["rougeL"])  )
    cider = np.mean( np.array(d["cider"])  )


    return bleu1,meteor,rougeL,cider

def bin_by_num(data,num):
    bin_list  = [0] # because 0 is needed in figure
    sd = sorted(data)
    cnt = 0
    for i,d in enumerate(sd):
        cnt+=1
        if cnt>=num or i == len(sd) - 1  :
            # bin_list.append(d+1)
            tmp  = round((d+1)/5)*5
            if tmp not in bin_list:
                bin_list.append(tmp )
            cnt = 0
    return bin_list



def plot_metric2node_num(node_num_list,metric_list,x_label,y_label_list,save_path,font_size,color_dict,fig_size,
                         calc_list,ylimit,b_loc,c_loc,line_icon ):
    # ymax = 4.5
    tmp_list = [{"bleu1":[],"meteor":[],"rougeL":[],"cider":[]}for _ in range(len(calc_list)-1)]
    for i in range(len(node_num_list)):
        for k in range(len(calc_list)-1):
            if calc_list[k] < node_num_list[i] <= calc_list[k+1]:
                tmp_list[k]["bleu1"].append(metric_list[i]["bleu1"])
                tmp_list[k]["meteor"].append(metric_list[i]["meteor"])
                tmp_list[k]["rougeL"].append(metric_list[i]["rougeL"])
                tmp_list[k]["cider"].append(metric_list[i]["cider"])
                break

    bleu1_list = []
    meteor_list = []
    rougeL_list = []
    cider_list = []
    for k in range(len(tmp_list)):
        bleu1,meteor,rougeL,cider = get_retr_metric(tmp_list[k])
        bleu1_list.append(bleu1)
        meteor_list.append(meteor)
        rougeL_list.append(rougeL)
        cider_list.append(cider)

    x = list(range(len(calc_list)))
    x.remove(0)

    wo_zero_calc_list_ori = copy.deepcopy(calc_list)
    wo_zero_calc_list_ori.remove(0)
    str_tick_list = [str(int(k)) for k in wo_zero_calc_list_ori]

    # plot_bleu_meteor_rougeL_cider(x,x_label,y_label,font_size,bleu1_list,meteor_list,rougeL_list,cider_list,str_tick_list,save_path,color_dict,ylimit)
    plot_2subfig_bleu_meteor_rougeL_cider(x,x_label,y_label_list,font_size,bleu1_list,meteor_list,rougeL_list,
                                          cider_list,str_tick_list,save_path,color_dict,b_loc=b_loc,c_loc=c_loc,
                                            line_icon=line_icon)


def load_json_by_line(path):
    data = []
    with open(path, "r", encoding="utf-8") as reader:
        # data = json.loads(reader.read())
        while True:
            line = reader.readline().strip()
            if line:
                dat = json.loads(line)
                data.append(dat)
            else:
                break
    return data

def plot_metric_vs_param(out_root_path,prefix,color_dict, path_input,ylimit,plot_list,bin_num=None,
                         y_label_list=None,line_icon=None ):

    fig_size = (9, 9)

    font_size = 20





    input_data = load_json_by_line(path_input)

    code_length_list =[]
    tree_node_num_list =[]
    cfg_node_num_list =[]
    comment_length_list =[]
    metric_list = []
    for i in range(len(input_data)):
        v = input_data[i ]
        code_length_list.append(int(v["tok_len"]))
        tree_node_num_list.append(int(v["ast_len"]))
        if 'cfg' in plot_list:
            cfg_node_num_list.append(int(v["cfg_node_num"]))
        comment_length_list.append(int(v["comment_len"]))
        metric_list.append( {"bleu1":v["bleu1"] ,"meteor":v["meteor"], "rougeL":v["rougel"],"cider":v["cider"]  } )

    print("max(code_length_list):{} min(code_length_list):{} ".format(max(code_length_list),min(code_length_list)))
    print("max(tree_node_num_list):{} min(tree_node_num_list):{} ".format(max(tree_node_num_list),min(tree_node_num_list)))
    print("max(comment_length_list):{} min(comment_length_list):{} ".format(max(comment_length_list),min(comment_length_list)))

    if 'tok' in plot_list:
        # code_length 0-130
        print("tok----")
        # c_loc = (0.747, 0.878)
        c_loc = (0.735, 0.878)
        # b_loc = (0.71 ,0.42 )
        b_loc = (0.698   ,0.42 )
        # calc_list = [0, 30, 50, 80, 120, 200, 1200]
        # calc_list = [0, 30, 50, 80,100, 120, 200, 1200]
        calc_list = bin_by_num(data=code_length_list, num=bin_num)
        print("calc_list: ",calc_list)
        save_path = os.path.join(out_root_path,"fig_toklen_score_"+ prefix+".pdf")
        # plot_metric2node_num(code_length_list,metric_list,x_label="Code length",y_label=y_label,save_path=save_path,
        #                      font_size=font_size,color_dict=color_dict,fig_size=fig_size,calc_list=list(np.linspace(0,100,11)),ylimit=ylimit)
        # plot_metric2node_num(code_length_list,metric_list,x_label="Token length",y_label_list=y_label_list,save_path=save_path,
        #                      font_size=font_size,color_dict=color_dict,fig_size=fig_size,
        #                     calc_list=[0,20,40,80,120,200,1200],ylimit=ylimit)
        # plot_metric2node_num(code_length_list,metric_list,x_label="Token length",y_label_list=y_label_list,save_path=save_path,
        #                      font_size=font_size,color_dict=color_dict,fig_size=fig_size,
        #                     calc_list=calc_list,ylimit=ylimit,b_loc=b_loc,c_loc=c_loc)
        plot_metric2node_num(code_length_list,metric_list,x_label='# code tokens',y_label_list=y_label_list,save_path=save_path,
                             font_size=font_size,color_dict=color_dict,fig_size=fig_size,
                            calc_list=calc_list,ylimit=ylimit,b_loc=b_loc,c_loc=c_loc,line_icon=line_icon)

        # 100 以后有部分数据不统计，因为统计进来如果分的太细了，就有的间隔里metric为0，如果分的太粗了，就有的间隔里metric太大
        # plot_metric2node_num(code_length_list,metric_list,x_label="Code length",y_label=y_label,save_path=save_path,
        #                      font_size=font_size,color_dict=color_dict,fig_size=fig_size,calc_list=list(np.linspace(0,100,11))+[130]) # 100往后的间隔metric太大
        # plot_metric2node_num(code_length_list,metric_list,x_label="Code length",y_label=y_label,save_path=save_path,
        #                      font_size=font_size,color_dict=color_dict,fig_size=fig_size,calc_list=list(np.linspace(0,130,14))) # 100往后的间隔里有的metric为0

    if 'ast' in  plot_list:
        # 0-3751
        print("ast----")
        c_loc = (0.747, 0.878)
        b_loc = (0.71 ,0.42 )
        # calc_list = [0,20,40,60,80,100,150]
        # calc_list = [0,30,50,70,90,150]
        calc_list = bin_by_num(data=tree_node_num_list, num=bin_num)
        print("calc_list: ", calc_list)
        save_path = os.path.join(out_root_path,"fig_treelen_score_"+prefix+".pdf")
        # plot_metric2node_num(tree_node_num_list,metric_list,x_label="AST node number",y_label=y_label,save_path=save_path,
        #                      font_size=font_size,color_dict=color_dict,fig_size=fig_size,calc_list=[0,10,20,30,40,50,60,70,80,90,100,560],ylimit=ylimit)
        # plot_metric2node_num(tree_node_num_list,metric_list,x_label="AST node number",y_label=y_label,save_path=save_path,
        #                      font_size=font_size,color_dict=color_dict,fig_size=fig_size,calc_list=[0,10,20,30,40,50,60,70,80,90,100,200],ylimit=ylimit)
        # plot_metric2node_num(tree_node_num_list,metric_list,x_label="AST node number",y_label=y_label,save_path=save_path,
        #                      font_size=font_size,color_dict=color_dict,fig_size=fig_size,calc_list=[0,20,40,60,80,100,200],ylimit=ylimit)
        # plot_metric2node_num(tree_node_num_list,metric_list,x_label="AST size",y_label_list=y_label_list,save_path=save_path,
        #                      font_size=font_size,color_dict=color_dict,fig_size=fig_size,
        #                      calc_list= calc_list ,ylimit=ylimit,b_loc=b_loc,c_loc=c_loc)
        plot_metric2node_num(tree_node_num_list,metric_list,x_label="# AST nodes",y_label_list=y_label_list,save_path=save_path,
                             font_size=font_size,color_dict=color_dict,fig_size=fig_size,
                             calc_list= calc_list ,ylimit=ylimit,b_loc=b_loc,c_loc=c_loc,line_icon=line_icon)

    if 'cfg' in plot_list:
        # 0-135
        print("cfg----")
        save_path = os.path.join(out_root_path,"fig_cfglen_score_"+prefix+".pdf")
        # plot_metric2node_num(cfg_node_num_list,metric_list,x_label="CFG node number",y_label=y_label,save_path=save_path,
        #                      font_size=font_size,color_dict=color_dict,fig_size=fig_size,calc_list=[0,5,10,15,20,25,30,35,40,45,50,150],ylimit=ylimit)
        # plot_metric2node_num(cfg_node_num_list,metric_list,x_label="CFG node number",y_label=y_label,save_path=save_path,
        #                      font_size=font_size,color_dict=color_dict,fig_size=fig_size,calc_list=[0,20,40,60,80,100,150],ylimit=ylimit)
        plot_metric2node_num(cfg_node_num_list,metric_list,x_label="CFG size",y_label_list=y_label_list,save_path=save_path,
                             font_size=font_size,color_dict=color_dict,fig_size=fig_size,calc_list=[0,20,40,60,80,100,150],ylimit=ylimit)
        # [0,5,10,15,20,25,30,35,40,45,50,55,60,150]
        # [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,150]

    if 'comment' in plot_list:
        # 0-30
        print("comment----")
        c_loc = (0.747, 0.878)
        b_loc = (0.71 ,0.42 )
        # calc_list = [0,5,10,15,20,25, 60]
        calc_list = [0,5,10,15,20,  55 ]
        # calc_list = bin_by_num(data=comment_length_list, num=bin_num)
        print("calc_list: ", calc_list)
        save_path = os.path.join(out_root_path,"fig_commentlen_score_"+prefix+".pdf")
        # plot_metric2node_num(comment_length_list,metric_list,x_label="Comment length",y_label=y_label,save_path=save_path,
        #                      font_size=font_size,color_dict=color_dict,fig_size=fig_size,calc_list=[0,5,10,15,20,25,30],ylimit=ylimit)
        # plot_metric2node_num(comment_length_list,metric_list,x_label="Comment length",y_label_list=y_label_list,save_path=save_path,
        #                      font_size=font_size,color_dict=color_dict,fig_size=fig_size,
        #                      calc_list=[0,5,10,15,20,25,30,60],ylimit=ylimit)
        # plot_metric2node_num(comment_length_list,metric_list,x_label="Comment length",y_label_list=y_label_list,save_path=save_path,
        #                      font_size=font_size,color_dict=color_dict,fig_size=fig_size,
        #                      calc_list=calc_list,ylimit=ylimit,b_loc=b_loc,c_loc=c_loc)
        plot_metric2node_num(comment_length_list,metric_list,x_label='# comment tokens',y_label_list=y_label_list,save_path=save_path,
                             font_size=font_size,color_dict=color_dict,fig_size=fig_size,
                             calc_list=calc_list,ylimit=ylimit,b_loc=b_loc,c_loc=c_loc,line_icon=line_icon)

if __name__ == "__main__":
    # python -m  vis.summarization.fse20_meta.fig_metric_vs_len

    prefix = "fse20_maml_"
    bin_num = 300

    ylimit = [0,4.5 ]
    plot_list = ['tok','ast','comment' ]
    y_label_list = ["Score", "Score"]

    b_m_r_c_color_dict = {"cider": "#FF6B6B",
                          "bleu1": "#6ED29B",
                          "rouge": "#76B7FD",
                          "meteor": "#F89406"
                          }

    # line_icon = { # issta20
    #     'cider': 'p',
    #     'bleu1':'>',
    #     'rouge': '.',
    #     'meteor': 'P'
    # }
    # line_icon = { # ase19
    #     'cider': '^',
    #     'bleu1':'*',
    #     'rouge': 'o',
    #     'meteor': 'D'
    # }
    line_icon = { # fse20
        'cider': 'o',
        'bleu1':'*',
        # 'bleu1':'H',
        'rouge': 'h',
        'meteor': 'd'
    }

    # path_input = '/data/wy/ghproj_d/fse20all/100_small/vis/p1_maml_202033230125.json'
    # path_input = '/data/wy/ghproj_d/fse20all/100_small/vis/p0_maml_202033230125.json'
    # out_root_path = '/data/wy/ghproj_d/fse20all/100_small/vis'


    # path_input = 'E:/research/dataset/nlp/fse20_smile/vis/202033230125/p1_maml.json'
    path_input = 'E:/research/dataset/nlp/fse20_smile/vis/202033230125/p0_maml.json'
    # out_root_path ='E:/research/dataset/nlp/fse20_smile/vis'
    out_root_path ='/'.join(path_input.split("/")[:-1])

    # out_root_path = os.path.join(out_root_path,path_input.split("/")[-2] )
    # if not os.path.exists(out_root_path):
    #     os.makedirs(out_root_path)

    prefix = prefix + path_input.split("/")[-1].split(".")[0]

    plot_metric_vs_param(out_root_path,prefix, b_m_r_c_color_dict,path_input,ylimit=ylimit,plot_list = plot_list ,
                         bin_num=bin_num,y_label_list=y_label_list,line_icon=line_icon)

    # color_dict={
    #     "BLEU-1":"gold",
    #     "METEOR":"green",
    #     "rougeL":"red",
    #     "CIDER":"blue"
    # }

# """
# 30 60 44 lv #1E3C2C
# 42 30 92 lan #2A1E5C
# 94 6 140 zi #5E068C
# """







