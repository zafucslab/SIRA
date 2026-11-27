from data_provider.data_loader import Dataset_Custom_Masked
from torch.utils.data import DataLoader

def data_provider(args, flag, pred_len=None):
    if pred_len is None:
        pred_len = args.pred_len
    Data = Dataset_Custom_Masked
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag in ('val', 'test'):
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    dataset_kwargs = dict(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[
            args.seq_len,
            args.label_len,
            pred_len,
        ],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
    )

    if args.data == 'custom_masked':
        dataset_kwargs['mask_data_path'] = getattr(
            args,
            'mask_data_path',
            None,
        )

    data_set = Data(**dataset_kwargs)
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader

