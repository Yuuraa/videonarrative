import argparse
import numpy as np
import os

from torch._C import Value

# from datautils import video_narr
from datautils import video_narr_bert

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='video-narr-bert', choices=['video-narr', 'video-narr-bert'], type=str)
    parser.add_argument('--answer_top', default=4000, type=int)
    parser.add_argument('--glove_pt',
                        help='glove pickle file, should be a map whose key are words and value are word vectors represented by numpy arrays. Only needed in train mode')
    parser.add_argument('--output_pt', type=str, default='/mnt/disk1/video_narr/{}/{}_{}_questions.pt')
    parser.add_argument('--vocab_json', type=str, default='/mnt/disk1/video_narr/{}/{}_vocab.json')
    parser.add_argument('--mode', default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--question_type', choices=['frameqa', 'action', 'transition', 'count', 'none'], default='none')
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--video_dir', default='/mnt/disk1/video_narr/raw_data', help='base directory of data' )

    args = parser.parse_args()
    np.random.seed(args.seed)

    # 본 경진대회는 "video-narr" 데이터셋 사용
    if args.dataset == 'video-narr':
        if not os.path.exists('/mnt/disk1/video_narr/{}'.format(args.dataset)):
            os.makedirs('/mnt/disk1/video_narr/{}'.format(args.dataset))
        video_narr.process_questions_mulchoices(args)

    elif args.dataset == 'video-narr-bert':
        if not os.path.exists('/mnt/disk1/video_narr/{}'.format(args.dataset)):
            os.makedirs('/mnt/disk1/video_narr/{}'.format(args.dataset))
        video_narr_bert.process_questions_mulchoices(args)
    
    else:
        raise ValueError(f"No such dataset defined: {args.dataset}")