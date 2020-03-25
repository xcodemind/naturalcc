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
import pandas as pd

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.switch_backend('agg')



def plot_somemodel_on_one_metric(x,x_label,y_label,metric ,font_size,y_dict,x_str_tick_list,save_path,color_dict,
                                 loc=None,    line_icon=None,m2modelname=None ):
    fig = plt.figure()


    # ymajorFormatter = FormatStrFormatter('%1.1f')  # 设置y轴标签文本的格式
    ymajorFormatter = FormatStrFormatter('%1.2f')  # 设置y轴标签文本的格式

    # gs = gridspec.GridSpec(2,1, height_ratios=[1,2])

    print("=====\nx:\n",x)
    # print("bleu1_list:\n",bleu1_list)
    # print("meteor_list:\n",meteor_list)
    # print("rouge_list:\n",rouge_list)
    # print("cider_list:\n",cider_list)

    # ax = plt.subplot(211)
    # ax = plt.subplot(gs[0])
    ax = plt.subplot(111)

    # ax.set_ylabel(y_label , fontsize=font_size)
    p_dict = {}
    # p_list  = []
    min_list  = []
    max_list = []
    for i in range(len(y_dict)) :
        for m , v in y_dict[i].items() :
           # p_dict[m] = ax.plot(x, v, marker=line_icon[m], linewidth=2.0, markersize=15, label=m, color=color_dict[m])
            p_dict[m], = ax.plot(x, v, marker=line_icon[m], linewidth=2.0, markersize=15 , color=color_dict[m])
            min_list.append(min(v))
            max_list.append(max(v))

    # ax.set_ylim( [min(min_list)-0.05, max(max_list)+0.2  ]  )  # 设置y轴刻度的范围
    # ax.set_ylim( [min(min_list)-0.05, max(max_list)+0.1  ]  )  # 设置y轴刻度的范围
    # ax.set_ylim( [min(min_list)-0.08, max(max_list)+0.06  ]  )  # 设置y轴刻度的范围
    if metric =='C':
        # ax.set_ylim( [min(min_list)-0.08, max(max_list)+0.02    ]  )  # 设置y轴刻度的范围
        ymin = max(0.01,min(min_list)-0.08)
        ymax = max(max_list)+0.02
        ax.set_ylim( [ymin , ymax]  )  # 设置y轴刻度的范围
    elif metric in ['B-1' , 'M']:
        ymin = max(0.01,min(min_list)-0.02)
        ymax = max(max_list)+0.01
        ax.set_ylim( [ ymin ,ymax ]  )  # 设置y轴刻度的范围
    else:
        ymin = max(0.01,min(min_list)-0.05)
        ymax = max(max_list)+0.01
        ax.set_ylim( [ ymin ,ymax ]  )  # 设置y轴刻度的范围

    # xmin = x[0]-4
    # xmax = x[-1]+4

    delta = x[-1]*0.04
    xmin = x[0] - delta
    xmax = x[-1] + delta
    ax.set_xlim([xmin, xmax])
    ax.tick_params(labelsize=font_size-1)
    print("x: ",x )
    ax.set_xticks(x)
    ax.set_xticklabels( x_str_tick_list)


    print("y_dict[0]: ",y_dict[0])
    print("y_dict[0].keys(): ",y_dict[0].keys())
    print("list(y_dict[0].keys())[0]: ",list(y_dict[0].keys())[0] )
    legend_list = [m2modelname[list(y_dict[0].keys())[0]],m2modelname[list(y_dict[1].keys())[0]],
                m2modelname[list(y_dict[2].keys())[0]],m2modelname[list(y_dict[3].keys())[0]]]
    print("legend_list: ",legend_list)
    fig.legend([p_dict[list(y_dict[0].keys())[0]] ,p_dict[list(y_dict[1].keys())[0]] ,
                p_dict[list(y_dict[2].keys())[0]] ,p_dict[list(y_dict[3].keys())[0]]] ,
               legend_list,
               fontsize=font_size-5,ncol=2 ,columnspacing=0.04,labelspacing=0.4,
               loc=   loc  ) #, loc=  'upper right'
    # fig.legend([p_dict[list(y_dict[0].keys())[0]][0],p_dict[list(y_dict[1].keys())[0]][0],
    #             p_dict[list(y_dict[2].keys())[0]][0],p_dict[list(y_dict[3].keys())[0]]][0],
    #            legend_list,
    #            fontsize=font_size-5,ncol=1,columnspacing=0.04,labelspacing=0.4,
    #            loc=   loc  ) #, loc=  'upper right'
    #
    # ax.set_xlabel(x_label, fontsize=font_size)
    # ax.set_ylabel(y_label , fontsize=font_size)
    # ax.yaxis.set_major_formatter(ymajorFormatter)

    # ax.grid()
    # xlist= [1,  20 , 40 ,60 ,80,100] # not plot grid line on 0.1
    if metric == 'C':
        grid_ylist=[0.1,0.2,0.3,0.4,0.5]
        yticklist=[0.1,0.2,0.3,0.4,0.5]

    elif metric == 'B-1':
        # grid_ylist = [0.1,0.12,0.15,0.18,0.20,0.23]
        grid_ylist =  [  0.18,0.20,0.22 ]
        yticklist = [ 0.16, 0.18,0.20,0.22,0.24]
    elif metric == 'M':
        # grid_ylist=[0.04,0.06,0.08,0.10]
        grid_ylist=[ 0.06,0.08,0.10]
        yticklist=[0.04,0.06,0.08,0.10,0.12]
    elif metric =='R-L':
        grid_ylist = [ 0.15,0.20]
        yticklist = [0.1,0.15,0.20,0.25]
    # ax = plot_grid(ax , xlist, grid_ylist, xmin=0, xmax=1, ymin=0, ymax=1, linewidth=None, linestyle=None, color='#B0B0B0')

    ax.set_yticks(yticklist)
    ax.set_yticklabels([str(k) for k in yticklist])

    grid_color =  '#B0B0B0'
    grid_linestyle = 'solid'
    grid_linewidth = 1
    # for dotx in xlist:
    for dotx in x:
        print("vline dotx:",dotx)
        print("ymin:{} ymax:{}".format(ymin,ymax  ))
        ax.vlines(ymin=ymin, ymax=ymax, x=dotx, color=grid_color, linestyle=grid_linestyle, linewidth=grid_linewidth)
        # ax.vlines(ymin=0,ymax=ymax, x=dotx, color='r' , linestyle='dotted',linewidth=4)
    for doty in grid_ylist:
        print("hline doty:",doty)
        ax.hlines(y=doty, xmin=xmin, xmax=xmax, color=grid_color, linestyle=grid_linestyle, linewidth=grid_linewidth)


    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_label , fontsize=font_size)
    ax.yaxis.set_major_formatter(ymajorFormatter)

    plt.tight_layout(w_pad=-5.1, rect=[0, 0, 1, 1])
    fig.savefig(save_path)

def plot_grid(axis,xlist,grid_ylist,xmin=0,xmax=1,ymin=0,ymax=1,linewidth=None,linestyle=None,color='#B0B0B0' ):
    for x in xlist:
        axis.vlines(ymin=ymin, ymax=ymax, x=x, color=color, linestyle=linestyle, linewidth=linewidth)
    for y in grid_ylist :
        axis.hlines(y=y , xmin=xmin, xmax=xmax ,   color=color, linestyle=linestyle,linewidth=linewidth)
    return axis

def plot_metric( metrics,portions ,save_path_root):
    m2modelname = {'Code2Seq': 'Code2Seq', 'XMetaL': 'XMetaL', 'MM2Seq': 'MM2Seq', 'Fine-Tuning': 'Fine-Tuning'}

    line_icon = { # fse20
        'Code2Seq': 'o',
        'XMetaL':'*',
        'MM2Seq': 'h',
        'Fine-Tuning': 'd'
    }

    color_dict = {"Code2Seq": "#FF6B6B",
                "XMetaL": "#6ED29B",
                "MM2Seq": "#76B7FD",
                "Fine-Tuning": "#F89406"}

    metric_dict={
        'B-1':'BLEU-1',
        'M':'METEOR',
        'R-L':'ROUGE-L',
        'C':'CIDER'

    }

    # x_label = 'Portion'
    x_label = 'Portion (%) of Ruby-Large dataset'
    fig_size = (9, 9)

    font_size = 20

    for metric in ['B-1' ,'M' ,'R-L' ,'C' ] :
        # y_dict = [{"Code2Seq":[]} ,{"XMetaL":[]},{"MM2Seq":[]} ,{"Fine-Tuning":[]}  ]
        code2seq = []
        maml = []
        mm2seq = []
        ft = []
        for p in portions:
            code2seq.append(metrics["Code2Seq"][metric][p])
            maml.append(metrics["XMetaL"][metric][p])
            mm2seq.append(metrics["MM2Seq"][metric][p])
            ft.append(metrics["Fine-Tuning"][metric][p])
            # y_dict["XMetaL"] = metrics["XMetaL"][metric]
            # y_dict["MM2Seq"] = metrics["MM2Seq"][metric]
            # y_dict["Fine-Tuning"] = metrics["Fine-Tuning"][metric]
        y_dict = [ {"XMetaL":maml},{"MM2Seq":mm2seq } ,{"Fine-Tuning":ft } ,
                   {"Code2Seq":code2seq}
                    ]

        x = [k/portions[0] for k in portions ]
        new = []
        for i in x :
            if i%1 == 0 :
                new.append(int(i))
        x = new
        x_str_tick_list = [str(k) for k in x]
        x = list(range(1,len(x_str_tick_list)+1))
        # loc = 'lower right'
        loc = (0.41 ,0.185)
        save_path = save_path_root+'_{}.pdf'.format(metric)
        plot_somemodel_on_one_metric(x=x,x_label=x_label,y_label=metric_dict[metric] ,metric=metric,font_size=font_size,y_dict=y_dict,
                                     x_str_tick_list=x_str_tick_list,save_path=save_path,color_dict=color_dict,
                                    loc= loc ,  line_icon=line_icon,m2modelname=m2modelname )

def get_ft_metric(path,cal_type='mean'):
    df = pd.read_excel(path )
    # print("df.shape:\n",df.shape )
    # print("df.columns \n",df.columns )
    metric={'B-1':{},'M':{},'R-L':{},'C':{}}
    for i in range(df.shape[0]):
        # print("i:{} type(df.iloc[i,0]): {}  df.iloc[i,0]:{} df.iloc[i]:{} ".format(i,type(df.iloc[i,0] ),df.iloc[i,0],df.iloc[i] ))
        if type(df.iloc[i,0]) == str and  '%' in df.iloc[i,0]:
            portion = float(df.iloc[i,0].split("(")[1].split("%")[0])/100
            print("portion: ",portion)
            for k,v in metric.items():
                if not v.__contains__(portion):
                    v[portion]  = []
                # print("df.iloc[i]['B-1']: ",df.iloc[i]['B-1'])
                # assert  False
                item = float(df.iloc[i][k])
                assert item != np.nan ,print("i:{} k:{} item:{}".format(i,k,item  ))
                v[portion].append(item)
                # assert len(v[portion]) <=5 , print("i:{} k:{} item:{} df.iloc[i,0]:{} v[portion]:{} ".format(i, k, item,df.iloc[i,0] ,v[portion] ))
                if len(v[portion]) > 0:
                    assert v[portion][-1] is not None and v[portion][-1] != np.nan , print("i:{} k:{} item:{} df.iloc[i,0]:{} v[portion]:{} ".format(i, k, item,df.iloc[i,0] ,v[portion] ))
                metric[k] = v
    # print("metric: \n", metric)
    for k ,v in metric.items():
        for kk,vv in v.items():
            if cal_type == 'mean':
                v[kk] = sum(vv)/float(len(vv))
            elif cal_type == 'max':
                v[kk] = max(vv)
            else:
                assert False
        metric[k] = v
    print("after cal, metric: \n", metric)
    return metric

if __name__ == '__main__':
    # 结果从腾讯文档复制到excel时，表头空缺列名必须要用内容填上，随便标注a b c d

    # portions = [0.01, 0.2 , 0.4 ,0.6 ,0.8,1.0]
    portions = [0.01,0.1, 0.2 , 0.4 ,0.6 ,0.8,1.0]

    cal_type = 'mean'
    # cal_type = 'max'

    # path_ft = 'E:/research/dataset/nlp/fse20_smile/vis/portion_ft_pretrain1epoch.xlsx'
    # path_code2seq = 'E:/research/dataset/nlp/fse20_smile/vis/portion_code2seq.xlsx'
    # path_mm2seq = 'E:/research/dataset/nlp/fse20_smile/vis/portion_tok8pathattnpointer.xlsx'
    # path_maml = 'E:/research/dataset/nlp/fse20_smile/vis/portion_maml.xlsx'

    # path_ft = 'E:/research/dataset/nlp/fse20_smile/vis/portion_ft_pretrain1epoch_add5all.xlsx' # 包含了go8java8javascript8php8python
    # path_code2seq = 'E:/research/dataset/nlp/fse20_smile/vis/portion_code2seq.xlsx'
    # path_mm2seq = 'E:/research/dataset/nlp/fse20_smile/vis/portion_tok8pathattnpointer_p0.01sc.xlsx' # p0.01的结果加了sc
    # path_maml = 'E:/research/dataset/nlp/fse20_smile/vis/portion_maml.xlsx'

    # path_ft = 'E:/research/dataset/nlp/fse20_smile/vis/portion_ft_pretrain1epoch.xlsx'
    # path_code2seq = 'E:/research/dataset/nlp/fse20_smile/vis/portion_code2seq.xlsx'
    # path_mm2seq = 'E:/research/dataset/nlp/fse20_smile/vis/portion_tok8pathattnpointer_p0.01sc.xlsx'  # p0.01的结果加了sc
    # path_maml = 'E:/research/dataset/nlp/fse20_smile/vis/portion_maml.xlsx'

    path_ft = 'E:/research/dataset/nlp/fse20_smile/vis/portion_ft_pretrain1epoch_addp0.1.xlsx'
    path_code2seq = 'E:/research/dataset/nlp/fse20_smile/vis/portion_code2seq_addp0.1.xlsx'
    path_mm2seq = 'E:/research/dataset/nlp/fse20_smile/vis/portion_tok8pathattnpointer_p0.01sc_addp0.1.xlsx'  # p0.01的结果加了sc
    path_maml = 'E:/research/dataset/nlp/fse20_smile/vis/portion_maml_addp0.1.xlsx'

    save_path_root = os.path.join('E:/research/dataset/nlp/fse20_smile/vis/' ,cal_type+"_"+"_".join(
        [k.split('/')[-1].split('.xlsx')[0] for k in [path_ft,path_code2seq,path_mm2seq,path_maml]]))

    os.makedirs(save_path_root,exist_ok=True )
    save_path_root = os.path.join(save_path_root,'portion_')

    # df=pd.read_excel( path1 )
    # # print("df:\n",df )
    # print("df.shape:\n",df.shape )
    # print("df.columns \n",df.columns )
    # print("df.columns.values\n",df.columns.values )

    # print("df.iloc[0,0] : ",df.iloc[0,0] )
    # print("df.iloc[0,1] : ",df.iloc[0,1] )
    # print("df.iloc[1,0] : ",df.iloc[1,0] )
    # print("df.iloc[1,1] : ",df.iloc[1,1] )
    metrics = {}
    metrics["Fine-Tuning"]  = get_ft_metric(path_ft,cal_type=cal_type)
    metrics["Code2Seq"]  = get_ft_metric(path_code2seq,cal_type=cal_type)
    metrics["MM2Seq"]  = get_ft_metric(path_mm2seq,cal_type=cal_type )
    metrics["XMetaL"]   = get_ft_metric(path_maml,cal_type=cal_type )


    plot_metric( metrics,portions ,save_path_root)


