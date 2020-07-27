import glob
import pandas as pd
from sklearn.model_selection import GroupKFold

import config


def make_folds(img_dir, out_path, n_fold=5):
    dataset = []
    for label, kind in enumerate(['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']):
        for path in glob.glob(f'{img_dir}/Cover/*.jpg'):
            dataset.append({
                'kind': kind,
                'image_name': path.split('/')[-1],
                'label': label
            })

    df = pd.DataFrame(dataset)
    df.loc[:, 'fold'] = 0
    
    gkf = GroupKFold(n_splits=n_fold)
    for i, (_, val_index) in enumerate(gkf.split(df.index, groups=df['image_name'])):
        df.loc[val_index, 'fold'] = i

    df.to_csv(out_path, index=False)


if __name__ == '__main__':
    make_folds(config.img_dir, config.folds_df_path)
