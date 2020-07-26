workdir = 'efficientnet-b4'
seed = 60

gpu = 0
device = 'cuda'

fold = 0
img_dir = '/home/data/alaska'
folds_df_path = '/home/baranov/alaska/train_folds.csv'
submission_path = '/home/baranov/alaska/submission.csv'

n_epochs = 150
batch_size = 16
num_workers = 4

model_name = 'efficientnet-b4'
n_classes = 4

loss = 'CrossEntropyLoss'

optimizer = dict(
    name='Adam',
    params=dict(
        lr=1e-6,
    )
)

scheduler = dict(
    name='CosineAnnealingWarmUpRestarts',
    params=dict(
        T_0=9,
        T_mult=1,
        eta_max=2.5e-4,
        T_up=1,
    )
)

apex = True
log_wandb = False
