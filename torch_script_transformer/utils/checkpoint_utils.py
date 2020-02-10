from __future__ import absolute_import, unicode_literals

import re
import os
import glob
import torch
import shutil
from collections import Mapping, Iterable
from torch_script_transformer.modules.transformer import TransformerModel


def to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, Mapping):
        return {k: to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, Iterable):
        return [to_cpu(e) for e in obj]
    else:
        return obj


def convert_to_fp16(obj):
    if isinstance(obj, torch.Tensor):
        if obj.dtype == torch.float32:
            return obj.astype(torch.float16)
        else:
            return obj
    elif isinstance(obj, Mapping):
        return {k: convert_to_fp16(v) for k, v in obj.items()}
    elif isinstance(obj, Iterable):
        return [convert_to_fp16(e) for e in obj]
    else:
        return obj


class CheckpointManager(object):

    def __init__(
        self,
        checkpoint_dir, model,
        keep_nb=None,
        save_as_fp16=False
    ):
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.checkpoint_dir = checkpoint_dir
        self.model = model
        self.keep_nb = keep_nb
        self.save_as_fp16 = save_as_fp16

        self.template = make_template(checkpoint_dir)
        self.checkpoint_files = list_all_checkpoint_files(checkpoint_dir)

    def save(self, step_nb):
        assert isinstance(step_nb, int)

        # Determine save path
        savepath = self.template.format(step_nb)

        # Add model to save obj
        saveobj = {
            'step_nb': step_nb,
            'model_args': self.model.args,
            'model_state_dict': to_cpu(self.model.state_dict())
        }
        if self.save_as_fp16:
            saveobj['model_state_dict'] = \
                convert_to_fp16(saveobj['model_state_dict'])

        # Save checkpoint
        torch.save(saveobj, savepath)
        shutil.copy(savepath, self.template.format('latest'))
        self.checkpoint_files = \
            list_all_checkpoint_files(self.checkpoint_dir)

        # Delete if needed
        if self.keep_nb:
            for filepath in self.checkpoint_files[:-self.keep_nb]:
                os.remove(filepath)
            self.checkpoint_files = \
                list_all_checkpoint_files(self.checkpoint_dir)


def make_template(checkpoint_dir):
    return os.path.join(checkpoint_dir, 'checkpoint-{}.pt')


def list_all_checkpoint_files(checkpoint_dir):
    template = make_template(checkpoint_dir)
    all_files = glob.glob(template.format('*'))
    file_pattern = re.compile(template.format('([0-9]+)'))
    all_files = list(filter(lambda x: file_pattern.match(x), all_files))
    all_files.sort(key=lambda x: int(file_pattern.match(x).group(1)))
    return all_files


def load_checkpoint(checkpoint_dir_or_file, src_dict, tgt_dict):
    if os.path.isfile(checkpoint_dir_or_file):
        checkpoint_file = checkpoint_dir_or_file
        checkpoint_dir = os.path.dirname(checkpoint_file)
    else:
        # Default to choosing latest
        checkpoint_dir = checkpoint_dir_or_file
        checkpoint_files = list_all_checkpoint_files(checkpoint_dir)
        assert len(checkpoint_files) > 0
        checkpoint_file = checkpoint_files[-1]

    savedobj = torch.load(checkpoint_file, map_location='cpu')
    step_nb = savedobj.get('step_nb', 0)

    model = TransformerModel.build_model(
        savedobj['model_args'], src_dict, tgt_dict)
    model.load_state_dict(savedobj['model_state_dict'])

    return model, step_nb
