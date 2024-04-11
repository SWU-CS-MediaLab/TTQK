import random
import shutil
import bisect

import numpy as np
from numpy import unique
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
import sys
from data_loader import *
from data_manager import *
from torchvision import transforms
from torch.autograd import Variable
import collections
import torch.nn.functional as F

def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model_best.pth.tar'))

def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of color image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, file_label 

def GenIdx( train_color_label, train_thermal_label):
    color_pos = collections.defaultdict(list)
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k,v in enumerate(train_color_label) if v==unique_label_color[i]]
        color_pos[unique_label_color[i]] = tmp_pos
        
    thermal_pos = collections.defaultdict(list)
    unique_label_thermal = np.unique(train_thermal_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k,v in enumerate(train_thermal_label) if v==unique_label_thermal[i]]
        thermal_pos[unique_label_thermal[i]] = tmp_pos
    return color_pos, thermal_pos

    
def GenCamIdx(gall_img, gall_label, mode):
    if mode =='indoor':
        camIdx = [1,2]
    else:
        camIdx = [1,2,4,5]
    gall_cam = []
    for i in range(len(gall_img)):
        gall_cam.append(int(gall_img[i][-10]))
    
    sample_pos = []
    unique_label = np.unique(gall_label)
    for i in range(len(unique_label)):
        for j in range(len(camIdx)):
            id_pos = [k for k,v in enumerate(gall_label) if v==unique_label[i] and gall_cam[k]==camIdx[j]]
            if id_pos:
                sample_pos.append(id_pos)
    return sample_pos
    
def ExtractCam(gall_img):
    gall_cam = []
    for i in range(len(gall_img)):
        cam_id = int(gall_img[i][-10])
        # if cam_id ==3:
            # cam_id = 2
        gall_cam.append(cam_id)
    
    return np.array(gall_cam)

def get_data(dataset,args,transform_train,transform_test):
    if dataset == "RegDB":
        root = osp.join(args.data_dir,"RegDB/")
        trainset = RegDBData(root,args.trial,transform_train)
        initset = RegDBData(root,args.trial,transform_test)
        query_img, query_label = process_test_regdb(root, trial=args.trial, modal='visible')
        gall_img, gall_label = process_test_regdb(root, trial=args.trial, modal='thermal')
        gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

    elif dataset == "SYSU-MM01":
        root = osp.join(args.data_dir,"SYSU-MM01/")
        trainset = SYSUData(root,transform_train)
        initset = SYSUData(root,transform_test)
        query_img, query_label, query_cam = process_query_sysu(root, mode=args.mode_sysu)
        gall_img, gall_label, gall_cam = process_gallery_sysu(root, mode=args.mode_sysu, trial=0)
        gallset = TestData(gall_img, gall_label,transform=transform_test, img_size=(args.img_w, args.img_h),test_cam=gall_cam)
        queryset = TestData(query_img, query_label, transform=transform_test,img_size=(args.img_w, args.img_h),test_cam=query_cam)

    elif dataset == "LLCM":
        root = osp.join(args.data_dir,"LLCM/")
        trainset = LLCMData(root,transform_train)
        initset = LLCMData(root,transform_test)
        if args.mode_llcm == 1:
            query_mode = 1
            gallary_mode = 2
        else:
            query_mode = 2
            gallary_mode = 1
        query_img, query_label, query_cam = process_query_llcm(root, mode=query_mode)
        gall_img, gall_label, gall_cam = process_gallery_llcm(root, mode=gallary_mode, trial=0)
        gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h),test_cam=gall_cam)
        queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h),test_cam=query_cam)

    elif dataset == "VCM":
        processed_data = VCM()
        #image-based method
        trainset = VideoDataset_train(processed_data.train_ir,processed_data.train_rgb,1,'video_train',transform_train)
        trainset.train_color_label = processed_data.rgb_label
        trainset.train_thermal_label = processed_data.ir_label
        initset = VideoDataset_train(processed_data.train_ir,processed_data.train_rgb,1,"video_train",transform_test)
        if args.mode_vcm == 1:
            gallset = VideoDataset_test(processed_data.gallery,1,"video_test",transform_test,processed_data.gallary_cam)
            queryset = VideoDataset_test(processed_data.query,1,"video_test",transform_test,processed_data.query_cam)
            gallset.test_label = processed_data.gallery_labels
            queryset.test_label = processed_data.query_labels
        else:
            gallset = VideoDataset_test(processed_data.gallery_1,1,"video_test",transform_test,processed_data.gallery_cam1)
            queryset = VideoDataset_test(processed_data.query_1,1,"video_test",transform_test,processed_data.query_cam1)
            gallset.test_label = processed_data.gallery_labels1
            queryset.test_label = processed_data.query_labels1
    else:
        print("The dataset name is wrong")
        exit(0)

    num_classes = len(unique(trainset.train_color_label))

    # testdataloader
    gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    sequentialSampler = SequentialSampler(trainset.train_color_label,trainset.train_thermal_label)

    initset.cIndex = sequentialSampler.index1
    initset.tIndex = sequentialSampler.index2

    init_loader = data.DataLoader(initset,batch_size=args.test_batch,shuffle=False,sampler=sequentialSampler,num_workers=args.workers)

    return trainset,num_classes,gall_loader,query_loader,init_loader

def extract_features(model, data_loader,training_phase = 1):
    features_all = []
    labels_all = []
    features_visible = []
    features_thermal = []
    model.eval()
    with torch.no_grad():
        for i, (input1,input2,labels1,labels2) in enumerate(data_loader):
            input1 = Variable(input1.cuda())
            input2 = Variable(input2.cuda())

            labels = torch.cat((labels1,labels2),dim = 0)
            labels = labels.cuda()

            features = model(input1,input2,training_phase=training_phase)["test"]

            features_v,features_t = features.chunk(2,0)

            for feature_v,pid in zip(features_v,labels1):
                features_visible.append((feature_v,pid))
                features_all.append(feature_v)
                labels_all.append(int(pid))

            for feature_t, pid in zip(features_t, labels2):
                features_thermal.append((feature_t,pid))
                features_all.append(feature_t)
                labels_all.append(int(pid))
    model.train()
    return features_all,labels_all,features_visible,features_thermal

def initial_classifier(model, data_loader):
    pid2features = collections.defaultdict(list)
    features_all, labels_all,_,_ = extract_features(model, data_loader)
    for feature, pid in zip(features_all, labels_all):
        pid2features[pid].append(feature)
    class_centers = [torch.stack(pid2features[pid]).mean(0) for pid in sorted(pid2features.keys())]
    class_centers = torch.stack(class_centers)
    return F.normalize(class_centers, dim=1).float().cuda()

def select_replay_samples(model,dataset,dataloader,training_phase=1, add_num=0, old_datas=None,
                          select_samples=2,pid_pos = None,replay_pids = None,pid_nums = None):
    replay_data = []

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256,128)),
        transforms.Pad(10),
        transforms.RandomCrop((256, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    color_imgs,thermal_imgs,color_labels,thermal_labels = dataset.train_color_image,dataset.train_thermal_image,dataset.train_color_label,dataset.train_thermal_label

    pid2color_imgs = collections.defaultdict(list)
    pid2thermal_imgs = collections.defaultdict(list)
    #
    for color_img, cpid in zip(color_imgs, color_labels):
        pid2color_imgs[cpid].append(color_img)
    for thermal_img,tpid in zip(thermal_imgs,thermal_labels):
        pid2thermal_imgs[tpid].append(thermal_img)

    pid2visible_features  = collections.defaultdict(list)
    pid2thermal_features = collections.defaultdict(list)

    #Select a few images that are closest to each ID center
    _ ,_ , features_visible, features_thermal = extract_features(model,dataloader,training_phase=training_phase)

    for feature_v ,pid in features_visible:
        pid2visible_features[int(pid)].append(feature_v)
    for feature_t, pid in features_thermal:
        pid2thermal_features[int(pid)].append(feature_t)


    class_centers_visible = [torch.stack(pid2visible_features[pid]).mean(0) for pid in sorted(pid2visible_features.keys())]
    class_centers_visible = F.normalize(torch.stack(class_centers_visible), dim=1)
    class_centers_thermal = [torch.stack(pid2thermal_features[pid]).mean(0) for pid in sorted(pid2thermal_features.keys())]
    class_centers_thermal = F.normalize(torch.stack(class_centers_thermal), dim=1)

    #Select 1/3 of the identities for each domain
    labels_all = list(set(color_labels))
    replay_pid_nums = len(labels_all)//3
    select_pids = np.random.choice(labels_all,replay_pid_nums, replace=False)

    if pid_pos is None:
        pid_pos = collections.defaultdict(int)
    idx = 0
    for pid in select_pids:
        visible_features = F.normalize(torch.stack(pid2visible_features[pid]), dim=1, p=2)
        thermal_features = F.normalize(torch.stack(pid2thermal_features[pid]), dim=1, p=2)
        visible_center = class_centers_visible[pid]
        thermal_center = class_centers_thermal[pid]
        simi_visible = torch.mm(visible_features, visible_center.unsqueeze(0).t())
        simi_thermal = torch.mm(thermal_features, thermal_center.unsqueeze(0).t())
        visible_idxs = torch.sort(simi_visible, dim=0)[1][:select_samples]
        thermal_idxs = torch.sort(simi_thermal,dim=0)[1][:select_samples]

        for color_idx,thermal_idx in zip(visible_idxs,thermal_idxs):
            color_idx = int(color_idx)
            thermal_idx = int(thermal_idx)
            thermal_nums = len(pid2thermal_imgs[pid])
            if thermal_idx >= thermal_nums:
                thermal_idx -= thermal_nums
            replay_data.append((pid2color_imgs[pid][color_idx],pid2thermal_imgs[pid][thermal_idx], pid+add_num,training_phase))
        if old_datas is None:
            pid_pos[pid] = idx
        else:
            pid_pos[pid+add_num] = idx + len(old_datas)
        idx += select_samples
    
    if replay_pids is None:
        replay_pids = select_pids.tolist()
        pid_nums = [replay_pid_nums]
    else:
        select_pids += add_num
        replay_pids.extend(select_pids.tolist())
        pid_nums.append(replay_pid_nums + pid_nums[-1])

    if old_datas is None:
        sampler = ReplaySampler(replay_data,pid_pos,replay_pids,training_phase,pid_nums,select_samples)
        data_loader_replay = data.DataLoader(ReplayData(replay_data,transform_train),
                             batch_size=32,num_workers=0,sampler = sampler,
                             pin_memory=False, drop_last=True)
    else:
        replay_data = old_datas + replay_data
        sampler = ReplaySampler(replay_data,pid_pos,replay_pids,training_phase,pid_nums,select_samples)
        #replay_data.append(old_datas)
        data_loader_replay = data.DataLoader(ReplayData(replay_data, transform_train),
                             batch_size=32,num_workers=0,sampler = sampler,
                             pin_memory=False, drop_last=True)

    return data_loader_replay,replay_data,pid_pos,replay_pids,pid_nums

class ReplaySampler(Sampler):
    def __init__(self, data_source, pid_pos, replay_pids, training_phase,pid_nums,select_samples):
        self.data_source = data_source
        self.pid_pos = pid_pos
        self.replay_pids = replay_pids
        self.training_phase = training_phase
        self.pid_nums = pid_nums
        self.select_samples = select_samples

    def __iter__(self):
        index = []
        domain_id = list(range(self.training_phase))
        random.shuffle(domain_id)
        for i in domain_id:
            if i == 0:
                random_ids = self.replay_pids[0:self.pid_nums[0]]
            else:
                random_ids = self.replay_pids[self.pid_nums[i-1]:self.pid_nums[i]]
            random.shuffle(random_ids)
            for id in random_ids:
                index.extend(list(range(self.pid_pos[id],self.pid_pos[id] + self.select_samples)))
        return iter(index)

    def __len__(self):
        return len(self.data_source)


class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """
    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, num_pos, batchSize):
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)
        self.num_pos = num_pos

        #print(sorted(color_pos.keys()))
        N = np.maximum(len(train_color_label), len(train_thermal_label))
        for j in range(int(N/(batchSize*num_pos))+1):
            batch_idx = np.random.choice(uni_label, batchSize, replace = False)
            #print(batch_idx)
            for i in range(batchSize):
                #print(batch_idx[i])
                sample_color  = np.random.choice(color_pos[batch_idx[i]], num_pos)
                sample_thermal = np.random.choice(thermal_pos[batch_idx[i]], num_pos)
                
                if j ==0 and i==0:
                    index1= sample_color
                    index2= sample_thermal
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))

        self.index1 = index1
        self.index2 = index2
        self.N = N
        
    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N

class SequentialSampler(Sampler):
    def __init__(self,train_color_label,train_thermal_label):
        super().__init__(train_color_label)
        color_len = len(train_color_label)
        thermal_len = len(train_thermal_label)
        index1 = list(range(color_len))
        index2 = list(range(thermal_len))

        if thermal_len < color_len:
            #index2.extend(np.random.choice(index2,color_len-thermal_len,replace=False))
            index2.extend(range(color_len-thermal_len))
        else:
            #index1.extend(np.random.choice(index1,thermal_len-color_len,replace=False))
            index1.extend(range(thermal_len-color_len))

        self.N = np.maximum(color_len,thermal_len)
        self.index1 = index1
        self.index2 = index2

    def __iter__(self):
        return iter(range(self.N))

    def __len__(self):
        return self.N

class AverageMeter(object):
    """Computes and stores the average and current value""" 
    def __init__(self):
        self.reset()
                   
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
 
def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise   
class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """  
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
            
def set_seed(seed, cuda=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def set_requires_grad(nets, requires_grad=False):
            """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
            Parameters:
                nets (network list)   -- a list of networks
                requires_grad (bool)  -- whether the networks require gradients or not
            """
            if not isinstance(nets, list):
                nets = [nets]
            for net in nets:
                if net is not None:
                    for param in net.parameters():
                        param.requires_grad = requires_grad