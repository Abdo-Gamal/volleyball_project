from torch.utils.data import DataLoader

def build_dataloader(
    dataset,
    batch_size,
    num_workers=2,
    collate_fn=None,
    shuffle=True,
    sampler=None,
    drop_last=False,
    prefetch_factor=1
):

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn,
        sampler=sampler,
        drop_last=drop_last,

    )
