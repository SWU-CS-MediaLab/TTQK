import torch.nn
from torch.backends import cudnn
from model import TTQK
from eval_metrics import eval_func2
from utils import *

import os.path as osp
import argparse

def main():
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker()

def main_worker():
    # modify this suffix according to yours
    suffix = f""

    save_model_dir = osp.join(args.save_model,suffix)
    if not osp.isdir(save_model_dir + '/'):
        os.makedirs(save_model_dir+'/')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_h,args.img_w)),
        transforms.Pad(10),
        transforms.RandomCrop((args.img_h, args.img_w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.img_h, args.img_w)),
            transforms.ToTensor(),
            normalize,
    ])

    datasets = ["RegDB","SYSU-MM01","LLCM","VCM"]
    dataset_regdb,num_classes_regdb,gallary_loader_regdb,\
            query_loader_regdb,init_loader_regdb = get_data(datasets[0],args,transform_train,transform_test)
    dataset_sysu,num_classes_sysu,gallary_loader_sysu,\
            query_loader_sysu,init_loader_sysu= get_data(datasets[1],args,transform_train,transform_test)
    dataset_llcm,num_classes_llcm,gallary_loader_llcm,\
            query_loader_llcm,init_loader_llcm= get_data(datasets[2],args,transform_train,transform_test)
    dataset_vcm,num_classes_vcm,query_loader_vcm,\
            gallary_loader_vcm,init_loader_vcm = get_data(datasets[3],args,transform_train,transform_test)

    net = TTQK(class_num=num_classes_regdb,embed_dim=768,KeyToken=True,GeneralToken=True,isShareBlock=False,instance_wise=True)
    net.cuda()
    net.eval()

    net.add_model(num_classes_sysu,init_loader_sysu)
    net.add_model(num_classes_llcm,init_loader_llcm)
    net.add_model(num_classes_vcm,init_loader_vcm)

    test_loaders = []
    test_loaders.append((gallary_loader_regdb, query_loader_regdb))
    test_loaders.append((gallary_loader_sysu, query_loader_sysu))
    test_loaders.append((gallary_loader_llcm, query_loader_llcm))
    test_loaders.append((gallary_loader_vcm, query_loader_vcm))

    model_path = osp.join(save_model_dir,args.resume)
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['state_dict'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))

    all_cmc = []
    all_mAP = []
    for name, test_loader in zip(datasets, test_loaders):
        cmc, mAP = eval_func2(net, name, test_loader, net.feat_dim, 4)
        all_cmc.append(cmc)
        all_mAP.append(mAP)

    for i in range(len(all_cmc)):
        print('Finished test! {} Rank-1: {:5.1%}, mAP: {:5.1%} '.format(datasets[i], all_cmc[i][0], all_mAP[i]))

    print("The incremental average accuracy：Rank-1: {:.3%}, mAP:{:.3%}".format(
        sum([cmc[0] for cmc in all_cmc]) / len(all_cmc), sum(all_mAP) / len(all_mAP)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continual learning for VI-ReID")
    # data
    parser.add_argument('-br', '--replay-batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--img_h', type=int, default=256, help="input height")
    parser.add_argument('--img_w', type=int, default=128, help="input width")
    parser.add_argument('--num_pos', type=int, default=4)
    parser.add_argument('--batch-size',type = int,default=8)
    parser.add_argument('--trial', default=1, type=int,
                        metavar='t', help='trial (only for RegDB dataset)')
    parser.add_argument('--test-batch', default=64, type=int,
                        metavar='tb', help='testing batch size')
    parser.add_argument('--mode-sysu',default="all",type = str,
                        help = "all or indoor(test mode for sysu)")
    parser.add_argument('--mode-llcm',default=1,type = int,
                        help = "test mode for llcm(for query)")
    parser.add_argument('--mode-vcm', default=1, type=int,
                        help="test mode for VCM(for query)")

    # resume path
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--seed', type=int, default=1)

    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join('/root/autodl-tmp/', ''))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--save-model',type = str,default=osp.join(working_dir,'save_model'))
    parser.add_argument('--rr-gpu', action='store_true',
                        help="use GPU for accelerating clustering")
    parser.add_argument('--vis-logs-dir',type = str,default=osp.join(working_dir,'vis_logs/'))
    args = parser.parse_args()
    main()