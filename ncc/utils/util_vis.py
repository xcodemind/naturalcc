# import torch

# import ncc.utils.constants as constants
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.switch_backend('agg')
# color_dict={
#     "bleu1":"#C563EB" ,
#     "meteor":"#5AC7E8",
#     "rouge":"#57FFA6" ,
#     "cider":"#EB805E"
# }
# "#daeef4" 浅蓝
# "#9bc5ef" 蓝
# "#295f3f" 深绿
# #5c1181 紫色
# "#66cc00" # 草绿

# #d55e00 # 朱红
# #e69f00 # 橙色
# #56b4e9 # 天蓝
# #cc79a7 # 紫红
# #f0e442 # 亮黄

# b_m_r_c_color_dict = {
#     "bleu1": "#5c1181",
#     "meteor": "#66cc00",  # "#bdffa9"
#     "rouge": "#9bc5ef",
#     "cider": "#295f3f"
# }

b_m_r_c_color_dict = {
    "bleu1": "#cc79a7", # 紫红
    "meteor": "#56b4e9", # 天蓝
    # "rouge": "#e69f00" ,# 橙色
    "rouge": "#f0e442" ,# 亮黄
    "cider":  "#d55e00" # 朱红
}



# class PT(object):
#     def __init__(self,pt_path,is_deepcom=False,test_pt_path_deepcom=None):
#         self.pt = torch.load(pt_path)
#         self.pt_code = self.pt["code"]
#         self.pt_comment = self.pt["comment"]
#         self.pt_cfg = self.pt["cfg"]
#         self.pt_tree = self.pt["tree"]
#         self.pt_code = self.pt["code"]
#         self.pt_key_list = self.pt["key_list"]
#         self.pt_data_len = len(self.pt_comment)
#         self.skip_list = []
#         if is_deepcom:
#             self.pt_deepcom = torch.load(test_pt_path_deepcom)
#             self.pt_deepcom_sbt_index =  self.pt_deepcom["sbt_tree_index"]
#
#
#     def is_equal(self,a,b):
#         len_a = len(a)
#         if len_a != len(b):
#
#             return False
#         else:
#             for i in range(len_a):
#                 if a[i] != "<unk>":
#                     if a[i] != b[i]:
#
#                         return False
#             return True
#
#     def get_index_in_ptfile(self,list_code2check,list_gts):
#         # print("===== code:\n{}\ngts:\n{}".format(code2check,gts ))
#         for i in range(self.pt_data_len):
#             if i not in self.skip_list:
#
#                 if self.is_equal(list_gts,self.pt_comment[i] ) :
#                     if self.is_equal(list_code2check,self.pt_code[i]):
#                         self.skip_list.append(i)
#                         return i
#         return None
#
#     def clean_pad_index(self,sent):
#         if constants.PAD in sent:
#             sent = sent[:sent.index(constants.PAD)]
#         return sent
#
#     def get_index_in_ptfile_sbt_index(self,list_sbt_index2check,list_gts):
#         # print("===== code:\n{}\ngts:\n{}".format(code2check,gts ))
#         list_sbt_index2check = self.clean_pad_index(list_sbt_index2check)
#         for i in range(self.pt_data_len):
#             if i not in self.skip_list:
#                 if self.is_equal(list_gts,self.pt_comment[i] ) :
#                     # print("list_sbt_index2check:\n{} self.pt_deepcom_sbt_index[i]:{} ".format(list_sbt_index2check, self.pt_deepcom_sbt_index[i]))
#                     if list_sbt_index2check == self.pt_deepcom_sbt_index[i] :
#                         self.skip_list.append(i)
#                         return i
#         return None



def plot_bleu_meteor_rougeL_cider(x,x_label,y_label,font_size,bleu1_list,meteor_list,rouge_list,cider_list,str_tick_list,save_path,color_dict,ylimit):
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.set_xlabel(x_label, fontsize=font_size)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=font_size)

    # color=color_dict["bleu1"]

    print("=====\nx:\n",x)
    print("bleu1_list:\n",bleu1_list)
    print("meteor_list:\n",meteor_list)
    print("rouge_list:\n",rouge_list)
    print("cider_list:\n",cider_list)
    max_v = max(max(bleu1_list),max(meteor_list),max(rouge_list),max(cider_list))
    print("max_v: ",max_v)
    if max_v > ylimit[-1]:
        assert False,print("max_v:{}  ylimit:{}".format(max_v,ylimit))


    ax.plot(x, bleu1_list, marker='>', linewidth=2.0, markersize=15, label='BLEU-1',color=color_dict["bleu1"])
    ax.plot(x, meteor_list, marker='P', linewidth=2.0, markersize=15, label='METEOR',color=color_dict["meteor"]) # "2"
    ax.plot(x, rouge_list, marker='.', linewidth=2.0, markersize=15, label='ROUGE',color=color_dict["rouge"])
    ax.plot(x, cider_list, marker='p', linewidth=2.0, markersize=15, label='CIDER',color=color_dict["cider"])

    # ax.legend(fontsize=font_size,ncol=2,loc=1) #, loc='upper center'
    ax.legend(fontsize=font_size,ncol=2,loc=7) #, loc='upper center'
    ax.tick_params(labelsize=font_size-1)

    ax.set_xticks(x)
    # ax.set_xticklabels([str(int(k)) for k in wo_zero_calc_list])
    ax.set_xticklabels( str_tick_list)
    # ax.set_ylim(0, 1.0)  # 设置y轴刻度的范围
    ax.set_ylim(*ylimit)  # 设置y轴刻度的范围

    plt.tight_layout(w_pad=-5.1, rect=[0, 0, 1, 1])
    fig.savefig(save_path)

def plot_2subfig_bleu_meteor_rougeL_cider(x,x_label,y_label_list,font_size,bleu1_list,meteor_list,rouge_list,cider_list,str_tick_list,save_path,color_dict,tag=None):
    fig = plt.figure()


    ymajorFormatter = FormatStrFormatter('%1.1f')  # 设置y轴标签文本的格式


    print("=====\nx:\n",x)
    print("bleu1_list:\n",bleu1_list)
    print("meteor_list:\n",meteor_list)
    print("rouge_list:\n",rouge_list)
    print("cider_list:\n",cider_list)

    ax = plt.subplot(211)

    if y_label_list[0] is not None:
        ax.set_ylabel(y_label_list[0], fontsize=font_size)

    pc = ax.plot(x, cider_list, marker='p', linewidth=2.0, markersize=15, label='C', color=color_dict["cider"])
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
        fig.legend([pc[0]], ["CIDER"], fontsize=font_size - 5, ncol=4,
                   loc=(0.76, 0.91 ))  # , loc='upper center'
    # fig.legend([pc[0]], ["C"], fontsize=font_size - 5, ncol=4,
    #            loc=(0.825, 0.885))  # , loc='upper center'
    # fig.legend([pc[0]], ["C"], fontsize=font_size - 5, ncol=4,
    #            loc= 3   )  # , loc='upper center'


    ax.yaxis.set_major_formatter(ymajorFormatter)
    ax.grid()

    ax = plt.subplot(212)

    if y_label_list[1] is not None:
        ax.set_ylabel(y_label_list[1], fontsize=font_size)
    pb = ax.plot(x, bleu1_list, marker='>', linewidth=2.0, markersize=15, label='B-1',color=color_dict["bleu1"])
    pm = ax.plot(x, meteor_list, marker='P', linewidth=2.0, markersize=15, label='M',color=color_dict["meteor"]) # "2"
    pr = ax.plot(x, rouge_list, marker='.', linewidth=2.0, markersize=15, label='R',color=color_dict["rouge"])
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

        fig.legend([pb[0],pm[0],pr[0]],["BLEU-1","METEOR","ROUGE" ],
                   fontsize=font_size-5,ncol=1,columnspacing=0.04,labelspacing=0.4,
                   loc=  (0.72,0.34 ) ) #, loc=  'upper right'

    ax.yaxis.set_major_formatter(ymajorFormatter)
    ax.grid()

    plt.tight_layout(w_pad=-5.1, rect=[0, 0, 1, 1])
    fig.savefig(save_path)

# def plot_2subfig_bleu_meteor_rouge_cider(x,x_label,y_label,font_size,bleu1_list,meteor_list,rouge_list,cider_list,str_tick_list,save_path,color_dict,ylimit):
#     fig = plt.figure()
#     ax = plt.subplot(211)
#     if y_label is not None:
#         ax.set_ylabel(y_label, fontsize=font_size)
#
#
#
#     print("=====\nx:\n",x)
#     print("bleu1_list:\n",bleu1_list)
#     print("meteor_list:\n",meteor_list)
#     print("rouge_list:\n",rouge_list)
#     print("cider_list:\n",cider_list)
#     # max_v = max(max(bleu1_list),max(meteor_list),max(rouge_list),max(cider_list))
#     # print("max_v: ",max_v)
#     # if max_v > ylimit[-1]:
#     #     assert False,print("max_v:{}  ylimit:{}".format(max_v,ylimit))
#
#
#     pb = ax.plot(x, bleu1_list, marker='>', linewidth=2.0, markersize=15, label='B-1',color=color_dict["bleu1"])
#     pm = ax.plot(x, meteor_list, marker='P', linewidth=2.0, markersize=15, label='M',color=color_dict["meteor"]) # "2"
#     pr = ax.plot(x, rouge_list, marker='.', linewidth=2.0, markersize=15, label='R',color=color_dict["rouge"])
#
#
#
#     ax.tick_params(labelsize=font_size-1)
#     ax.set_xticks(x)
#     ax.set_xticklabels( str_tick_list)
#     ax.set_ylim( [min(min(bleu1_list),min(meteor_list),min(rouge_list))-0.5, max(max(bleu1_list),max(meteor_list),max(rouge_list))+0.5 ]  )  # 设置y轴刻度的范围
#
#     ax = plt.subplot(212)
#     if y_label is not None:
#         ax.set_ylabel(y_label, fontsize=font_size)
#     ax.set_xlabel(x_label, fontsize=font_size)
#
#     pc = ax.plot(x, cider_list, marker='p', linewidth=2.0, markersize=15, label='C', color=color_dict["cider"])
#
#     ax.tick_params(labelsize=font_size-1)
#     ax.set_xticks(x)
#     ax.set_xticklabels( str_tick_list)
#     ax.set_ylim([min(cider_list)-0.5,max(cider_list)+0.5 ])  # 设置y轴刻度的范围
#
#     # ax.legend(fontsize=font_size,ncol=2,loc=1) #, loc='upper center'
#     # ax.legend([pb,pm,pr,pc],[lb,lm,lr,lc],fontsize=font_size,ncol=2,loc=7) #, loc='upper center'
#     # fig.legend([pb[0],pm[0],pr[0],pc[0]],["B-1","M","R","C" ],fontsize=font_size-5,ncol=4,loc=1) #, loc='upper center'
#     fig.legend([pb[0],pm[0],pr[0],pc[0]],["B-1","M","R","C" ],fontsize=font_size-5,ncol=4,loc=(0.2,0.88   )) #, loc='upper center'
#
#     plt.tight_layout(w_pad=-5.1, rect=[0, 0, 1, 1])
#     fig.savefig(save_path)

def plot_bar_bleu_meteor_rouge_cider(x,x_label,y_label,font_size,bleu1_list,meteor_list,rouge_list,cider_list,str_tick_list,save_path,color_dict,ylimit):
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.set_xlabel(x_label, fontsize=font_size)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=font_size)


    print("=====\nx:\n",x)
    print("bleu1_list:\n",bleu1_list)
    print("meteor_list:\n",meteor_list)
    print("rouge_list:\n",rouge_list)
    print("cider_list:\n",cider_list)
    max_v = max(max(bleu1_list),max(meteor_list),max(rouge_list),max(cider_list))
    print("max_v: ",max_v)
    if max_v > ylimit[-1]:
        assert False,print("max_v:{}  ylimit:{}".format(max_v,ylimit))


    # ax.plot(x, bleu1_list, marker='>', linewidth=2.0, markersize=15, label='BLEU-1',color=color_dict["bleu1"])
    # ax.plot(x, meteor_list, marker='P', linewidth=2.0, markersize=15, label='METEOR',color=color_dict["meteor"]) # "2"
    # ax.plot(x, rouge_list, marker='.', linewidth=2.0, markersize=15, label='ROUGE',color=color_dict["rouge"])
    # ax.plot(x, cider_list, marker='p', linewidth=2.0, markersize=15, label='CIDER',color=color_dict["cider"])

####
    name_list = ['0%', '20%', '40%', '60%','80%','100%']
    x = list(range(len(bleu1_list)))
    total_width, n = 0.8, 4
    width = total_width / n

    plt.bar(x, meteor_list, width=width, label='METEOR', fc=color_dict["meteor"])
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, bleu1_list, width=width, label='BLEU-1', fc=color_dict["bleu1"])
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, rouge_list, width=width, label='ROUGE',  fc=color_dict["rouge"])
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, cider_list, width=width, label='CIDER', tick_label=name_list, fc=color_dict["cider"])
#####
    # ax.legend(fontsize=font_size,ncol=2,loc=7) #, loc='upper center'
    ax.legend(fontsize=font_size,ncol=2,loc=1) #, loc='upper center'
    ax.tick_params(labelsize=font_size-1)

    # ax.set_xticks(x)
    # ax.set_xticklabels([str(int(k)) for k in wo_zero_calc_list])
    # ax.set_xticklabels( str_tick_list)
    # ax.set_ylim(0, 1.0)  # 设置y轴刻度的范围
    ax.set_ylim(*ylimit)  # 设置y轴刻度的范围

    plt.tight_layout(w_pad=-5.1, rect=[0, 0, 1, 1])
    fig.savefig(save_path)


def plot_bar(x_tick_list,y,x_label,y_label,font_size,color,bar_width,ylimit,save_path):
    fig = plt.figure()
    ax = plt.subplot(111)
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=font_size)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=font_size)

    x = list(range(len(y)))

    plt.bar(x, y, width=bar_width, tick_label=x_tick_list ,fc= color)

    for a, b in zip(x, y):
        if b > 0.9999     :
            plt.text(a, b + 0.05, '%.6e' % b, ha='center', va='bottom', fontsize=font_size-1  )
        else:
            plt.text(a, b + 0.05, '%.g' % b, ha='center', va='bottom', fontsize=font_size-1  )

    # ax.legend(fontsize=font_size, ncol=2, loc=1)  # , loc='upper center'
    ax.tick_params(labelsize=font_size - 1)

    ax.set_ylim(ylimit)  # 设置y轴刻度的范围

    plt.tight_layout(w_pad=-5.1, rect=[0, 0, 1, 1])
    fig.savefig(save_path)

def plot_bar_subplot(x_tick_list,y_list,x_label_list,y_label_list,font_size,color,bar_width,ylimit,save_path):
    fig = plt.figure()
    for i in range(len(y_list)):
        x_tick = x_tick_list[i]
        y = y_list[i]
        x_label = x_label_list[i]
        y_label = y_label_list[i]
        ax = plt.subplot(len(y_list),1,i+1)
        if x_label is not None:
            ax.set_xlabel(x_label, fontsize=font_size-3)
        if y_label is not None:
            ax.set_ylabel(y_label, fontsize=font_size)

        x = list(range(len(y)))

        plt.bar(x, y, width=bar_width, tick_label=x_tick ,fc= color)

        for a, b in zip(x, y):
            if b > 0.9999     :
                plt.text(a, b + 0.2, '%.6e' % b, ha='center', va='bottom', fontsize=font_size-5  )
            else:
                plt.text(a, b + 0.2 , '%.g' % b, ha='center', va='bottom', fontsize=font_size-5    )

        # ax.legend(fontsize=font_size, ncol=2, loc=1)  # , loc='upper center'
        ax.tick_params(labelsize=font_size - 1)

        ax.set_ylim(ylimit)  # 设置y轴刻度的范围

    plt.tight_layout(w_pad=-5.1, rect=[0, 0, 1, 1])
    fig.savefig(save_path)

def get_floatnum_from_log(tag,li):
    try:
        a = li[li.index(tag)+len(tag):]
    except:
        print("tag:{}   li:\n{}".format(tag,li))
        assert False
    # print("a: ",a)
    # print("type(a): ",type(a))
    return float(a.split(",")[0].strip())

# def get_num_from_log(tag,li):
#     try:
#         a = li[li.index(tag)+len(tag):]
#     except:
#         print("tag:{}   li:\n{}".format(tag,li))
#         assert False
#     # print("a: ",a)
#     # print("type(a): ",type(a))
#     return float(a.split(",")[0].strip())

def plot_diff_y(x,y1,y2,xlabel,ylabel1,ylabel2,font_size,save_path,color_list,ylimit1,ytick1,ylimit2,ytick2,
                cut_x_position,color_dict,linewidth=4.0 ):
    import matplotlib
    matplotlib.rcParams['figure.figsize'] = [10, 3.6]  # for square canvas
    matplotlib.rcParams['figure.subplot.left'] = 0
    matplotlib.rcParams['figure.subplot.bottom'] = 0
    matplotlib.rcParams['figure.subplot.right'] = 1
    matplotlib.rcParams['figure.subplot.top'] = 1

    # fig_main = plt.figure(figsize=[20,10 ])
    fig, left_axis = plt.subplots()
    # fig.subplots_adjust(right_axis=0,75)

    right_axis = left_axis.twinx()

    p1, = left_axis.plot(x, y1, color=color_list[0],linewidth=linewidth)
    p2, = right_axis.plot(x, y2, color=color_list[1],linewidth=linewidth)

    left_axis.set_xlim(0, 112)
    left_axis.set_xticks([0,10,20,30,40,50,60,70,80,90,100,110])

    left_axis.set_ylim(ylimit1)
    left_axis.set_yticks(ytick1)

    right_axis.set_ylim(ylimit2)
    right_axis.set_yticks(ytick2)

    left_axis.set_xlabel(xlabel, fontsize=font_size)
    left_axis.set_ylabel(ylabel1, fontsize=font_size)
    right_axis.set_ylabel(ylabel2, fontsize=font_size)

    left_axis.yaxis.label.set_color(p1.get_color())
    right_axis.yaxis.label.set_color(p2.get_color())

    # tkw = dict(size=font_size, width=0.5,length=0.5)
    # tkw = dict(size=font_size)
    # left_axis.tick_params(axis='y', colors=p1.get_color(), **tkw)
    # right_axis.tick_params(axis='y', colors=p2.get_color(), **tkw)
    left_axis.tick_params(axis='y', colors=p1.get_color() , labelsize=font_size)
    right_axis.tick_params(axis='y', colors=p2.get_color() , labelsize=font_size)
    left_axis.tick_params(axis='x' , labelsize=font_size)

    # plt.hlines(y=max(y), xmin=min(x), xmax=max(x),     color='r', linestyle='dotted')
    left_axis.vlines(ymin=0,ymax=ylimit1[-1], x=cut_x_position, color=color_dict["bleu1"], linestyle='dotted',linewidth=linewidth)

    # left_axis.vlines(ymin=0,ymax=ylimit1[-1], x=106, color=color_dict["bleu1"], linestyle='dotted')

    # left_axis.annotate("",
    #             xy=(0, y1[cut_x_position-1]), xycoords='data',
    #             xytext=(cut_x_position, y1[cut_x_position-1]), textcoords='data',
    #             arrowprops=dict(arrowstyle="<->",
    #                             connectionstyle="arc3", color=color_dict["bleu1"], lw=2),
    #             )
    # left_axis.annotate("",
    #             xy=(cut_x_position, y1[cut_x_position-1]), xycoords='data',
    #             xytext=(x[-1], y1[cut_x_position-1]), textcoords='data',
    #             arrowprops=dict(arrowstyle="<->",
    #                             connectionstyle="arc3", color=color_dict["bleu1"], lw=2),
    #             )

    # plt.text(1.2 * min(x), max(y), 'Range of residuals: %g' % (max(y) - min(y)),
    #          rotation=90, fontsize=16)

    plt.savefig(save_path)

    plt.tight_layout(w_pad=-5.1, rect=[0, 0, 1, 1])
    fig.savefig(save_path)
    # fig_main.savefig(save_path)

