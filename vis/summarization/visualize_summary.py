
import os, sys, argparse
from collections import OrderedDict
import itertools

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--article')
    parser.add_argument('--filename')
    parser.add_argument('--sysout')
    parser.add_argument('--output_html_dir')

    return parser.parse_args()


def parse_sysout(sysout_file):
    fin = open(sysout_file, encoding='utf8')
    N = int(fin.readline())
    idx2lbl = {}
    lbl2idx = {}
    for i in range(N):
        line = fin.readline()
        fds = line.strip().split()
        idx = int(fds[1])
        idx2lbl[idx] = fds[0]
        lbl2idx[fds[0]] = idx
    # print(idx2lbl)
    # print(lbl2idx)
    article_probs = []
    for line in fin:
        if line.startswith('Predicted Distri:'):
            fds = line.strip().split('\t')
            probs = fds[1].strip().split('|')
            probs = [list(map(float, prob.strip().split())) for prob in probs]
            '''
            # print( probs )
            print( '**********************' )
            print( probs )
            '''
            probs = [p[ lbl2idx['T'] ] for p in probs]
            # print( probs )
            article_probs.append( probs )

    return article_probs


def show_html(cur_fname, cur_articles, cur_probs, out_dir):
    label = cur_fname.strip().split('/')[1]
    html_file = os.path.join( out_dir, label + '.html' )
    topk_each_seg = 3
    topk_doc = 10

    def get_rank(cur_prob):
        idxs = list(range(len(cur_prob)))
        idxs.sort(key=lambda k: -cur_prob[k])

        return idxs

    with open(html_file, 'w', encoding='utf8') as fout:
        body = []
        for i in range(len(cur_articles)):
            cur_article = cur_articles[i]
            cur_prob = cur_probs[i]
            cnt = 0
            '''
            idxs = list(range(len(cur_prob)))
            idxs.sort(key=lambda k: -cur_prob[k])
            '''
            idxs = get_rank(cur_prob)
            for sent, score in zip(cur_article, cur_prob):
                if cnt in idxs[0:topk_each_seg]:
                    body.append( '<p><mark>{} &nbsp; &nbsp; &nbsp; &nbsp; {}</mark></p>\n'.format(sent, score) )
                else:
                    body.append( '<p>{} &nbsp; &nbsp; &nbsp; &nbsp; {}</p>\n'.format(sent, score) )
                cnt += 1
            body.append('<hr><br>')

        # get summary
        all_articles = list(itertools.chain.from_iterable(cur_articles))
        all_probs = list(itertools.chain.from_iterable(cur_probs))
        print('len', len(all_articles))
        print('len', len(all_probs))
        idxs = get_rank(all_probs)
        sum_idxs = idxs[0:topk_doc]
        sum_idxs.sort()
        sum_list = []
        for i in sum_idxs:
            sum_list.append( '<p>{} &nbsp; &nbsp; &nbsp; &nbsp; {}</p>\n'.format(all_articles[i], all_probs[i]) )

        html = '''
<html>
    <head><title>{fname}</title></head>
    <body>
        <h1>{fname}</h1>
        {body}
        <h1>Summaries</h1>
        {sum}
    </body>
</html>
        '''.format(fname=label, body=''.join(body), sum=''.join(sum_list))
        fout.write(html)


def show(article_file, fname_file, sysout_file, out_dir):
    fnames = open(fname_file, encoding='utf8').readlines()
    articles = []
    for line in open(article_file):
        article = line.strip().split(' <S_SEP> ')
        articles.append(article)
    print( len(fnames) )
    print( len(articles) )
    article_probs = parse_sysout(sysout_file)
    print( len(article_probs) )

    cur_fname = fnames[0]
    cur_articles = []
    cur_probs = []
    for i in range(len(fnames)):
        fname = fnames[i]
        article = articles[i]
        probs = article_probs[i]
        if fname == cur_fname:
            cur_articles.append( article )
            cur_probs.append( probs )
        else:
            print( len( cur_articles ) )
            print( len( cur_probs ) )
            print( cur_fname )
            show_html(cur_fname, cur_articles, cur_probs, out_dir)

            cur_fname = fname
            cur_articles = [article]
            cur_probs = [probs]

    if len(cur_articles) > 0:
        print( len( cur_articles ) )
        print( len( cur_probs ) )
        print( cur_fname )
        show_html(cur_fname, cur_articles, cur_probs, out_dir)


if __name__ == '__main__':
    args = get_args()
    print(args)
    if not os.path.exists(args.output_html_dir):
        os.mkdir(args.output_html_dir)
    show(args.article, args.filename, args.sysout, args.output_html_dir)
