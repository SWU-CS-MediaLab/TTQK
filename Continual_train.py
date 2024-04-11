from __future__ import absolute_import,print_function
import argparse
import copy

import torch.nn
from torch.backends import cudnn
from utils import *
from model import TTQK
from trainer import Trainer
from eval_metrics import eval_func2
from optimizer import adamw,sgd
from scheduler import create_scheduler
from tensorboardX import SummaryWriter

def main():
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker()

def main_worker():

    cudnn.benchmark = True
    print("==========\nArgs:{}\n==========".format(args))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # modify this suffix according to yours
    suffix = f""
    log_name = "log_" + suffix + ".txt"
    sys.stdout = Logger(osp.join(args.logs_dir, log_name))

    save_model_dir = osp.join(args.save_model,suffix)
    if not osp.isdir(save_model_dir + '/'):
        os.makedirs(save_model_dir+'/')

    vis_log_dir = osp.join(args.vis_logs_dir,suffix, "stag1/")
    if not os.path.isdir(vis_log_dir):
        os.makedirs(vis_log_dir)
    writer = SummaryWriter(vis_log_dir)

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

    #change the training order by modifying the order of datasets below
    datasets = ["RegDB","SYSU-MM01","LLCM","VCM"]

    #Create data and dataloaders
    dataset_regdb,num_classes_regdb,gallary_loader_regdb,\
        query_loader_regdb,init_loader_regdb = get_data(datasets[0],args,transform_train,transform_test)

    ngallary = len(gallary_loader_regdb.dataset.test_label)
    nquery = len(query_loader_regdb.dataset.test_label)
    
    print('Dataset {} statistics:'.format(datasets[0]))
    print('  ------------------------------')
    print('  subset   | # ids | # images')
    print('  ------------------------------')
    print('  visible  | {:5d} | {:8d}'.format(num_classes_regdb, len(dataset_regdb.train_color_label)))
    print('  thermal  | {:5d} | {:8d}'.format(num_classes_regdb, len(dataset_regdb.train_thermal_label)))
    print('  ------------------------------')
    print('  query    | {:5d} | {:8d}'.format(len(unique(query_loader_regdb.dataset.test_label)),nquery))
    print('  query    | {:5d} | {:8d}'.format(len(unique(gallary_loader_regdb.dataset.test_label)),ngallary))
    print('  ------------------------------')

    #Create model
    start_epoch = 0
    net = TTQK(class_num=num_classes_regdb,embed_dim=768,KeyToken=True,GeneralToken=True,isShareBlock=False,instance_wise=True)

    if args.step == 1 and args.resume != "":
        model_path = osp.join(save_model_dir,args.resume)
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            print('==> loaded checkpoint {} (epoch {})'
                  .format(args.resume, checkpoint['epoch']))
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    net.to(device)

    #train settings
    names = ["regdb"]
    test_loaders = [(gallary_loader_regdb,query_loader_regdb)]

    # optimizer
    if args.optimizer == "sgd":
        optimizer = sgd(args,net,1)
    elif args.optimizer == "adam":
        optimizer = adamw(args,net,1)
        scheduler = create_scheduler(optimizer)

    #store the results of every stage
    all_cmc_perstage = collections.defaultdict(list)
    all_mAP_perstage = collections.defaultdict(list)

    # Start training
    print('Continual training starts!')
    trainer = Trainer(net)
    color_pos, thermal_pos = GenIdx(dataset_regdb.train_color_label, dataset_regdb.train_thermal_label)

    for epoch in range(start_epoch,15):
        if args.optimizer == "sgd":
            current_lr = adjust_learning_rate(optimizer,epoch)
        else:
            scheduler.step(epoch)
            current_lr = optimizer.param_groups[0]["lr"]

        sampler = IdentitySampler(dataset_regdb.train_color_label,dataset_regdb.train_thermal_label,color_pos,thermal_pos,
                                  num_pos=args.num_pos,batchSize=args.batch_size)
        dataset_regdb.cIndex = sampler.index1
        dataset_regdb.tIndex = sampler.index2
        train_loader_regdb = data.DataLoader(dataset_regdb, batch_size=args.batch_size*args.num_pos, num_workers=0,
                                             sampler = sampler,drop_last = True)

        trainer.train(epoch,train_loader_regdb,None,optimizer,1,0,writer = writer)

        #save the best model
        if epoch > 0 and epoch % 2 == 0:
            for name, test_loader in zip(datasets, test_loaders):
                cmc, mAP = eval_func2(net,name, test_loader,net.feat_dim,1,epoch)
                print(name + ' Rank-1: {:.2%}| mAP: {:.2%} '.format(cmc[0],mAP))

        #evaluate in the last epoch
        if (epoch == 15 - 1):
            all_cmc = []
            all_mAP = []
            for name, test_loader in zip(datasets, test_loaders):
                cmc, mAP_regdb = eval_func2(net,name, test_loader,net.feat_dim,1,epoch)
                all_cmc.append(cmc)
                all_mAP.append(mAP_regdb)
            save_checkpoint({
                'state_dict': net.state_dict(),
                'epoch': epoch + 1,
                'mAP': mAP_regdb,
            }, True, fpath=osp.join(save_model_dir, 'working_checkpoint_step_1.pth.tar'))

            all_cmc_perstage["stag1"].append(all_cmc)
            all_mAP_perstage["stag1"].append(all_mAP)
            print('Finished epoch {:3d}  RegDB Rank-1: {:5.1%}, mAP: {:5.1%} '.format(epoch, cmc[0],mAP_regdb))

        writer.add_scalar("lr",current_lr,epoch)
    #

    replay_dataloader, regdb_replay_dataset, pid_pos, replay_pids,pid_nums = select_replay_samples(net, dataset_regdb,init_loader_regdb, training_phase=1,select_samples=4)


    del train_loader_regdb, dataset_regdb, optimizer, trainer

    # start to train next dataset
    dataset_sysu,num_classes_sysu,gallary_loader_sysu,\
        query_loader_sysu,init_loader_sysu= get_data(datasets[1],args,transform_train,transform_test)

    nquery = len(query_loader_sysu.dataset.test_label)
    ngallary = len(gallary_loader_sysu.dataset.test_label)

    print('Dataset {} statistics:'.format(datasets[1]))
    print('  ------------------------------')
    print('  subset   | # ids | # images')
    print('  ------------------------------')
    print('  visible  | {:5d} | {:8d}'.format(num_classes_sysu, len(dataset_sysu.train_color_label)))
    print('  thermal  | {:5d} | {:8d}'.format(num_classes_sysu, len(dataset_sysu.train_thermal_label)))
    print('  ------------------------------')
    print('  query    | {:5d} | {:8d}'.format(len(unique(query_loader_sysu.dataset.test_label)),nquery))
    print('  gallery  | {:5d} | {:8d}'.format(len(unique(gallary_loader_sysu.dataset.test_label)),ngallary))
    print('  ------------------------------')
    
    if args.step == 2 and args.resume !="":
        model_path = osp.join(save_model_dir, args.resume)
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            start_epoch = 0
            net.load_state_dict(checkpoint['state_dict'])
            print('==> loaded checkpoint {} (epoch {})'
                  .format(args.resume, checkpoint['epoch']))
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    # add expandable specific tokens and SAB
    net.add_model(num_classes_sysu,init_loader_sysu)
    add_num = sum(net.class_per_task[:-1])

    # Create old frozen model
    old_model = copy.deepcopy(net)
    old_model = old_model.cuda()
    
    names.append("sysu")
    test_loaders.append((gallary_loader_sysu,query_loader_sysu))

    #Re-initialize the optimizer
    if args.optimizer == "sgd":
        optimizer = sgd(args,net,2)
    elif args.optimizer == "adam":
        optimizer = adamw(args,net,2)
        scheduler = create_scheduler(optimizer)

    vis_log_dir = osp.join(args.vis_logs_dir, suffix ,"stag2/")
    if not os.path.isdir(vis_log_dir):
        os.makedirs(vis_log_dir)

    writer = SummaryWriter(vis_log_dir)

    trainer = Trainer(net)
    color_pos, thermal_pos = GenIdx(dataset_sysu.train_color_label, dataset_sysu.train_thermal_label)
    for epoch in range(start_epoch, 20):
        if args.optimizer == "sgd":
            current_lr = adjust_learning_rate(optimizer, epoch)
        else:
            scheduler.step(epoch)
            current_lr = optimizer.param_groups[0]["lr"]

        sampler = IdentitySampler(dataset_sysu.train_color_label, dataset_sysu.train_thermal_label, color_pos,
                                  thermal_pos, num_pos=args.num_pos, batchSize=args.batch_size)
        dataset_sysu.cIndex = sampler.index1
        dataset_sysu.tIndex = sampler.index2
        train_loader_sysu = data.DataLoader(dataset_sysu, batch_size=args.batch_size * args.num_pos, num_workers=0,
                                             sampler=sampler, drop_last=True)

        trainer.train(epoch, train_loader_sysu, replay_dataloader, optimizer, training_phase=2,
                        add_num=add_num, old_model=old_model, replay=True,writer = writer)

        if epoch > 0 and epoch % 2 == 0:
            all_cmc = []
            all_mAP = []
            for name, test_loader in zip(datasets, test_loaders):
                cmc, mAP = eval_func2(net,name, test_loader,net.feat_dim,2,epoch)
                all_cmc.append(cmc[0])
                all_mAP.append(mAP)
                print(name + ' Rank-1: {:.2%}| mAP: {:.2%} '.format(cmc[0],mAP))
            print("current incremental average accuracy：Rank-1: {:.3%}, mAP:{:.3%}".format(sum(all_cmc)/len(all_cmc),sum(all_mAP) / len(all_mAP)))


        if (epoch == 20 - 1):
            all_cmc = []
            all_mAP = []
            for name, test_loader in zip(datasets, test_loaders):
                cmc, mAP_sysu= eval_func2(net,name,test_loader,net.feat_dim,2,epoch)
                all_cmc.append(cmc)
                all_mAP.append(mAP_sysu)

            save_checkpoint({
                'state_dict': net.state_dict(),
                'epoch': epoch + 1,
                'mAP': mAP_sysu,
            }, True, fpath=osp.join(save_model_dir, 'working_checkpoint_step_2.pth.tar'))
            all_cmc_perstage["stag2"].append(all_cmc)
            all_mAP_perstage["stag2"].append(all_mAP)
            print('Finished epoch {:3d}  SYSU Rank-1: {:5.1%}, mAP: {:5.1%} '.format(epoch, cmc[0],mAP_sysu))

            for i in range(len(all_cmc)-1):
                print('Finished stag2! {} Rank-1: {:5.1%}, mAP: {:5.1%} '.format(datasets[i], all_cmc[i][0],all_mAP[i]))

            print("Stage2's incremental average accuracy：Rank-1: {:.3%}, mAP:{:.3%}".format(sum([cmc[0] for cmc in all_cmc])/len(all_cmc),sum(all_mAP)/len(all_mAP)))


        writer.add_scalar("lr",current_lr,epoch)

    replay_dataloader, sysu_replay_dataset, pid_pos, replay_pids,pid_nums = select_replay_samples(net, dataset_sysu,init_loader_sysu,training_phase=2,
                                                                     add_num=num_classes_regdb,
                                                                     old_datas=regdb_replay_dataset,
                                                                     select_samples= 4,
                                                                     pid_pos=pid_pos,
                                                                     replay_pids = replay_pids,
                                                                     pid_nums = pid_nums)

    del train_loader_sysu,trainer,optimizer,init_loader_sysu,old_model

    # start to train next dataset
    dataset_llcm,num_classes_llcm,gallary_loader_llcm,\
        query_loader_llcm,init_loader_llcm= get_data("LLCM",args,transform_train,transform_test)
    
    nquery = len(query_loader_llcm.dataset.test_label)
    ngallary = len(gallary_loader_llcm.dataset.test_label)

    print('Dataset {} statistics:'.format(datasets[2]))
    print('  ------------------------------')
    print('  subset   | # ids | # images')
    print('  ------------------------------')
    print('  visible  | {:5d} | {:8d}'.format(num_classes_llcm, len(dataset_llcm.train_color_label)))
    print('  thermal  | {:5d} | {:8d}'.format(num_classes_llcm, len(dataset_llcm.train_thermal_label)))
    print('  ------------------------------')
    print('  query    | {:5d} | {:8d}'.format(len(unique(query_loader_llcm.dataset.test_label)),nquery))
    print('  gallery  | {:5d} | {:8d}'.format(len(unique(gallary_loader_llcm.dataset.test_label)),ngallary))
    print('  ------------------------------')
    
    if args.step == 3 and args.resume !="":
        model_path = osp.join(save_model_dir, args.resume)
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            start_epoch = 0
            net.load_state_dict(checkpoint['state_dict'])
            print('==> loaded checkpoint {} (epoch {})'
                  .format(args.resume, checkpoint['epoch']))
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    # add expandable specific tokens and SAB
    net.add_model(num_classes_llcm,init_loader_llcm)
    add_num = sum(net.class_per_task[:-1])

    # Create old frozen model
    old_model = copy.deepcopy(net)
    old_model = old_model.cuda()

    names.append("llcm")
    test_loaders.append((gallary_loader_llcm, query_loader_llcm))

    # Re-initialize the optimizer
    if args.optimizer == "sgd":
        optimizer = sgd(args,net,3)
    elif args.optimizer == "adam":
        optimizer = adamw(args,net,3)
        scheduler = create_scheduler(optimizer)

    vis_log_dir = osp.join(args.vis_logs_dir,suffix,"stag3/")
    if not os.path.isdir(vis_log_dir):
        os.makedirs(vis_log_dir)
    writer = SummaryWriter(vis_log_dir)

    trainer = Trainer(net)
    color_pos, thermal_pos = GenIdx(dataset_llcm.train_color_label, dataset_llcm.train_thermal_label)
    for epoch in range(start_epoch, 20):
        if args.optimizer == "sgd":
            current_lr = adjust_learning_rate(optimizer, epoch)
        else:
            scheduler.step(epoch)
            current_lr = optimizer.param_groups[0]["lr"]

        sampler = IdentitySampler(dataset_llcm.train_color_label, dataset_llcm.train_thermal_label, color_pos,
                                  thermal_pos, num_pos=args.num_pos, batchSize=args.batch_size)
        dataset_llcm.cIndex = sampler.index1
        dataset_llcm.tIndex = sampler.index2
        train_loader_llcm = data.DataLoader(dataset_llcm, batch_size=args.batch_size * args.num_pos, num_workers=0,
                                             sampler=sampler, drop_last=True)

        trainer.train(epoch, train_loader_llcm, replay_dataloader, optimizer, training_phase=3,
                      add_num=add_num, old_model=old_model, replay=True,writer = writer)

        if epoch > 0 and epoch % 2 == 0:
            all_cmc = []
            all_mAP = []
            for name, test_loader in zip(datasets, test_loaders):
                cmc, mAP = eval_func2(net, name, test_loader, net.feat_dim,3,epoch)
                all_cmc.append(cmc[0])
                all_mAP.append(mAP)
                print(name + ' Rank-1: {:.2%}| mAP: {:.2%} '.format(cmc[0], mAP))
            print("current incremental average accuracy：Rank-1: {:.3%}, mAP:{:.3%}".format(sum(all_cmc)/len(all_cmc),sum(all_mAP) / len(all_mAP)))


        if (epoch == 20 - 1):
            all_cmc = []
            all_mAP = []
            for name, test_loader in zip(datasets, test_loaders):
                cmc, mAP_llcm = eval_func2(net, name, test_loader,net.feat_dim,3,epoch)
                all_cmc.append(cmc)
                all_mAP.append(mAP_llcm)

            save_checkpoint({
                'state_dict': net.state_dict(),
                'epoch': epoch + 1,
                'mAP': mAP_llcm,
            }, True, fpath=osp.join(save_model_dir, 'working_checkpoint_step_3.pth.tar'))
            all_cmc_perstage["stag3"].append(all_cmc)
            all_mAP_perstage["stag3"].append(all_mAP)
            print('Finished epoch {:3d}  LLCM Rank-1: {:5.1%}, mAP: {:5.1%} '.format(epoch, cmc[0], mAP_llcm))

            for i in range(len(all_cmc) - 1):
                print(
                    'Finished stag3! {} Rank-1: {:5.1%}, mAP: {:5.1%} '.format(datasets[i], all_cmc[i][0], all_mAP[i]))

            print("Stage3's incremental average accuracy：Rank-1: {:.3%}, mAP:{:.3%}".format(sum([cmc[0] for cmc in all_cmc])/len(all_cmc),sum(all_mAP)/len(all_mAP)))

        writer.add_scalar("lr",current_lr,epoch)

    # Select replay data of llcm
    replay_dataloader, llcm_replay_dataset,pid_pos,replay_pids,pid_nums = select_replay_samples(net, dataset_llcm,init_loader_llcm, training_phase=3,
                                                                   add_num=add_num,
                                                                   old_datas=sysu_replay_dataset,
                                                                   select_samples=4,
                                                                   pid_pos=pid_pos,
                                                                   replay_pids = replay_pids,
                                                                   pid_nums=pid_nums)

    del train_loader_llcm, trainer, optimizer, init_loader_llcm,dataset_llcm,old_model

    dataset_vcm,num_classes_vcm,query_loader_vcm,\
        gallary_loader_vcm,init_loader_vcm = get_data("VCM",args,transform_train,transform_test)

    net.add_model(num_classes_vcm, init_loader_vcm)
    add_num = sum(net.class_per_task[:-1])

    if args.step == 4 and args.resume != "":
        model_path = osp.join(save_model_dir, args.resume)
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            start_epoch = 29
            net.load_state_dict(checkpoint['state_dict'])
            print('==> loaded checkpoint {} (epoch {})'
                  .format(args.resume, checkpoint['epoch']))
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    # Create old frozen model
    old_model = copy.deepcopy(net)
    old_model = old_model.cuda()

    names.append("vcm")
    test_loaders.append((gallary_loader_vcm, query_loader_vcm))

    # Re-initialize the optimizer
    if args.optimizer == "sgd":
        optimizer = sgd(args,net,4)
    elif args.optimizer == "adam":
        optimizer = adamw(args,net,4)
        scheduler = create_scheduler(optimizer)

    vis_log_dir = osp.join(args.vis_logs_dir,suffix, "stag4/")
    if not os.path.isdir(vis_log_dir):
        os.makedirs(vis_log_dir)
    writer = SummaryWriter(vis_log_dir)

    trainer = Trainer(net)
    color_pos, thermal_pos = GenIdx(dataset_vcm.train_color_label, dataset_vcm.train_thermal_label)
    for epoch in range(start_epoch, 30):
        if args.optimizer == "sgd":
            current_lr = adjust_learning_rate(optimizer, epoch)
        else:
            scheduler.step(epoch)
            current_lr = optimizer.param_groups[0]["lr"]

        sampler = IdentitySampler(dataset_vcm.train_color_label, dataset_vcm.train_thermal_label, color_pos,
                                  thermal_pos, num_pos=args.num_pos, batchSize=args.batch_size)
        dataset_vcm.cIndex = sampler.index1
        dataset_vcm.tIndex = sampler.index2
        train_loader_vcm = data.DataLoader(dataset_vcm, batch_size=args.batch_size * args.num_pos, num_workers=0,
                                             sampler=sampler, drop_last=True)

        trainer.train(epoch, train_loader_vcm, replay_dataloader, optimizer, training_phase=4,
                      add_num=add_num, old_model=old_model, replay=True,writer = writer)

        if epoch > 0 and epoch % 2 == 0:
            all_cmc = []
            all_mAP = []
            for name, test_loader in zip(datasets, test_loaders):
                cmc, mAP = eval_func2(net, name, test_loader, net.feat_dim,4,epoch)
                all_cmc.append(cmc[0])
                all_mAP.append(mAP)
                print(name + ' Rank-1: {:.2%}| mAP: {:.2%} '.format(cmc[0], mAP))
            print("current incremental average accuracy：Rank-1: {:.3%}, mAP:{:.3%}".format(sum(all_cmc)/len(all_cmc),sum(all_mAP) / len(all_mAP)))
        
        if (epoch == 30 - 1):
            all_cmc = []
            all_mAP = []
            for name, test_loader in zip(datasets, test_loaders):
                cmc, mAP_vcm = eval_func2(net, name, test_loader,net.feat_dim,4,epoch)
                all_cmc.append(cmc)
                all_mAP.append(mAP_vcm)

            save_checkpoint({
                'state_dict': net.state_dict(),
                'epoch': epoch + 1,
                'mAP': mAP_vcm,
            }, True, fpath=osp.join(save_model_dir, 'working_checkpoint_step_4.pth.tar'))
            all_cmc_perstage["stag4"].append(all_cmc)
            all_mAP_perstage["stag4"].append(all_mAP)
            print('Finished epoch {:3d}  VCM Rank-1: {:5.1%}, mAP: {:5.1%} '.format(epoch, cmc[0], mAP_vcm))

            for i in range(len(all_cmc) - 1):
                print('Finished stag4! {} Rank-1: {:5.1%}, mAP: {:5.1%} '.format(datasets[i], all_cmc[i][0], all_mAP[i]))

            print("Stage4's incremental average accuracy：Rank-1: {:.3%}, mAP:{:.3%}".format(sum([cmc[0] for cmc in all_cmc])/len(all_cmc),sum(all_mAP)/len(all_mAP)))

        writer.add_scalar("lr",current_lr,epoch)

    del train_loader_vcm, trainer, optimizer, init_loader_vcm, dataset_vcm,old_model

    save_checkpoint({
        'state_dict':net.state_dict(),
        'all_cmc_perstage':all_cmc_perstage,
        'all_map_perstage':all_mAP_perstage
    },False,fpath = osp.join(save_model_dir,'last_stage_checkpoint.tar'))



def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        lr = args.lr
    elif epoch >= 20 and epoch < 50:
        lr = 0.1 * args.lr
    elif epoch >= 50:
        lr = args.lr * 0.01

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return lr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continual learning for VI-ReID")
    # data
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--img_h', type=int, default=256, help="input height")
    parser.add_argument('--img_w', type=int, default=128, help="input width")
    parser.add_argument('--num_pos', type=int, default=4)
    parser.add_argument('--batch-size',type = int,default=8)
    parser.add_argument('--test-batch', default=64, type=int,
                        metavar='tb', help='testing batch size')
    parser.add_argument('--trial', default=1, type=int,
                        metavar='t', help='trial (only for RegDB dataset)')
    parser.add_argument('--mode-sysu',default="all",type = str,
                        help = "all or indoor(test mode for sysu)")
    parser.add_argument('--mode-llcm',default=1,type = int,
                        help = "test mode for llcm(for query)")
    parser.add_argument('--mode-vcm', default=1, type=int,
                        help="test mode for VCM(for query)")

    # optimizer
    parser.add_argument('--optimizer',type = str,default="adam")
    parser.add_argument('--weight-decay', type=float, default=5e-4)

    #SGD
    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--momentum', type=float, default=0.9)

    #Adam
    parser.add_argument('--base_lr', type=int, default=5e-4)
    parser.add_argument('--lr_pretrain', type=int, default=0.5)


    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--step',type = int,default = 0)
    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--print-freq', type=int, default=200)
    parser.add_argument('--margin', type=float, default=0.3, help='margin for the triplet loss with batch hard')

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