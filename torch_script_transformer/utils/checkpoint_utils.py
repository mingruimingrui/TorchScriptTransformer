import os
import torch


def save_model(
    model,
    checkpoint_dir,
    postfix=None
):
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    basename = os.path.join(checkpoint_dir, 'checkpoint')
    if postfix is not None:
        basename += '-{}'.format(postfix)
    filepath = '{}.pt'.format(basename)

    save_obj = (model.args, model.state_dict())
    torch.save(save_obj, filepath)


# def load_model():
