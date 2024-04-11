import torch

def adamw(args, net,training_phase):
    lr = args.base_lr
    if training_phase > 1 and training_phase <=3:
        lr = 0.1 * lr
    weight_decay = args.weight_decay

    ignored_params = list(map(id, net.sabs.parameters()))\
                     +list(map(id,net.patchembed.proj.parameters()))

    if net.base == False:
        if training_phase > 1:
            if net.isShareBlock:
                ignored_params += list(map(id, net.task_tokens[:training_phase - 1].parameters()))
            else:
                ignored_params += list(map(id, net.task_tokens[:training_phase - 1].parameters()))\
                                 +list(map(id,net.tabs[:training_phase - 1].parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    params = [{"params": base_params, 'lr': lr},
              {'params': net.sabs.parameters(), 'lr': lr * args.lr_pretrain},
              {'params': net.patchembed.proj.parameters(), 'lr': lr * args.lr_pretrain}]

    if net.base == False:
        if training_phase > 1:
            if net.isShareBlock:
                params += [{"params": net.task_tokens[:training_phase - 1].parameters(), 'lr': lr * 0.1}]
            else:
                params += [{"params": net.task_tokens[:training_phase - 1].parameters(), 'lr': lr * 0.1},
                           {"params": net.tabs[:training_phase - 1].parameters(), 'lr': lr * 0.1}]

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    return optimizer

def sgd(args, net,training_phase):
    lr = args.lr
    if training_phase > 1 and training_phase <= 3:
        lr = 0.1 * lr
    weight_decay = args.weight_decay

    if net.use_backbone:
        ignored_params = list(map(id, net.backbone.parameters()))
    else:
        ignored_params = list(map(id, net.sabs.parameters()))\
                        +list(map(id,net.patchembed.proj.parameters()))

    if net.base == False:
        if training_phase > 1:
            if net.ModalitySpecific:
                ignored_params += list(map(id, net.task_tokens_v[:training_phase - 1].parameters()))\
                                + list(map(id, net.task_tokens_t[training_phase - 1].parameters()))
            else:
                if net.isShareBlock:
                    ignored_params += list(map(id, net.task_tokens[:training_phase - 1].parameters()))
                else:
                    ignored_params += list(map(id, net.task_tokens[:training_phase - 1].parameters()))\
                                     +list(map(id,net.tabs[:training_phase-1].parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    if net.use_backbone:
        params = [{"params": base_params, 'lr': lr},
                  {"params":net.backbone.parameters(),"lr": lr * 0.1}]
    else:
        params = [{"params": base_params, 'lr': lr},
                  {'params': net.sabs.parameters(), 'lr': lr * args.lr_pretrain},
                  {'params': net.patchembed.proj.parameters(), 'lr': lr * args.lr_pretrain}]

    if net.base == False:
        if training_phase > 1:
            if net.ModalitySpecific:
                params += [{"params": net.task_tokens_v[:training_phase - 1].parameters(), 'lr': lr * 0.1},
                           {"params":net.task_tokens_t[:training_phase-1].parameters(), 'lr': lr * 0.1}]
            else:
                if net.isShareBlock:
                    params += [{"params": net.task_tokens[:training_phase - 1].parameters(), 'lr': lr * 0.1}]
                else:
                    params += [{"params": net.task_tokens[:training_phase - 1].parameters(), 'lr': lr * 0.1},
                               {"params": net.tabs[:training_phase - 1].parameters(), 'lr': lr * 0.1}]

    optimizer = torch.optim.SGD(params, lr=lr, momentum = 0.9,weight_decay=weight_decay,nesterov=True)

    return optimizer