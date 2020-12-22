class efficientPipeline:
    model_name="efficientPipeline"
    batch_size = 32
    WORKERS = 1
    classes = 9
    epochs = 1
    optimizer = "torch.optim.AdamW"
    optimizer_parm = {'lr':1e-3,'weight_decay':0.00001}
    scheduler = "torch.optim.lr_scheduler.CosineAnnealingLR"
    scheduler_parm = {'T_max':5500,'eta_min':0.000001}
    loss_fn = 'torch.nn.SmoothL1Loss'
