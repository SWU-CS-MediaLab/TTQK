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