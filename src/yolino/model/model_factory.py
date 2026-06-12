# Copyright 2023 Karlsruhe Institute of Technology, Institute for Measurement
# and Control Systems
#
# This file is part of YOLinO.
#
# YOLinO is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# YOLinO is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# YOLinO. If not, see <https://www.gnu.org/licenses/>.
#
# ---------------------------------------------------------------------------- #
# ----------------------------- COPYRIGHT ------------------------------------ #
# ---------------------------------------------------------------------------- #
from copy import copy
from datetime import datetime

import torch
from yolino.model.yolino_net import YolinoNet
from yolino.utils.enums import Network
from yolino.utils.logger import Log


def load_checkpoint(args, dataset_specs, load_best=False, allow_failure=True):
    model = get_model(args, dataset_specs)

    model.to(args.cuda)
    epoch = 0

    # TODO read further on https://pytorch.org/tutorials/beginner/dist_overview.html#data-parallel-training
    # model = torch.nn.parallel.DistributedDataParallel(model) 

    scheduler_checkpoint = {}
    source_id = "unknown"
    if not args.retrain:
        try:
            model, scheduler_checkpoint, epoch, source_id = get_from_checkpoint(model, args, load_best=load_best)
        except (FileNotFoundError, ValueError) as fnf_error:
            if allow_failure:
                Log.warning('No pretrained weights, starting training from scratch, because %s' % fnf_error)
                epoch = 0
            else:
                raise fnf_error

    # If the checkpoint came from a *different* run (i.e. warm-start of a new
    # experiment from another run's weights, e.g. exp60 loading exp19 ep42 as
    # trunk init), treat it as init-only: keep the loaded weights but reset
    # epoch counter and discard the foreign run's scheduler/optimizer state.
    if epoch != 0 and source_id != "unknown" and source_id != args.id:
        Log.warning(
            "Checkpoint run ID=%s differs from current run ID=%s; "
            "treating as init-only (epoch reset 0, scheduler state dropped)." %
            (source_id, args.id)
        )
        epoch = 0
        scheduler_checkpoint = {}

    if epoch == 0:
        Log.debug("Start training from scratch")
    else:
        Log.warning('Resume from checkpoint at epoch %d (checkpoint run ID=%s; current run ID=%s)' %
                    (epoch, source_id, args.id))
    return model, scheduler_checkpoint, epoch


def get_model_specs(args):
    model_class = get_model_class(args)
    return model_class.specs


def get_model_class(args):
    if args.model == Network.YOLO_CLASS:
        model_class = YolinoNet
    else:
        raise NotImplementedError("We could not find the model_class %s" % args.model)
    return model_class


def get_model(args, coords):
    model_class = get_model_class(args)
    Log.debug("Load model %s" % model_class)
    model = model_class(args=args, coords=coords)
    return model


def get_from_checkpoint(model, args, load_best):
    checkpoint = get_checkpoint(args, load_best=load_best)

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    epoch = checkpoint['epoch']

    source_id = checkpoint.get('ID', 'unknown')

    if "scheduler_state_dict" in checkpoint:
        scheduler_checkpoint = checkpoint['scheduler_state_dict']
    else:
        scheduler_checkpoint = {}

    return model, scheduler_checkpoint, epoch, source_id


def print_checkpoint(checkpoint):
    for k in checkpoint.keys():
        if k == "args":
            from pprint import pprint
            pprint(checkpoint["args"].__dict__)
        else:
            Log.debug("We do not print %s. Enable in model_factory.py" % k)


def get_checkpoint(args, print_debug=False, check_args=True, load_best=False):
    Log.debug("\nTry loading pretrain weights from %s" % (str(args.paths.pretrain_model)))
    if load_best and args.explicit_model is None:
        checkpoint = torch.load(str(args.paths.pretrain_best_model), map_location=torch.device(args.cuda))
    else:
        checkpoint = torch.load(str(args.paths.pretrain_model), map_location=torch.device(args.cuda))

    if print_debug:
        print_checkpoint(checkpoint)

    if check_args:
        relevant_keys = ['dataset', 'img_size', 'model', 'darknet_cfg', 'linerep', 'num_predictors', 'activations',
                         'training_variables', 'scale']

        if "args" not in checkpoint:
            raise ValueError("Your checkpoint is outdated as it does not have any args assigned. "
                             "Please run a training and try again.")

        args_diff = {}
        for k in relevant_keys:
            if k == "darknet_cfg":
                import os
                if os.path.basename(args.darknet_cfg) != os.path.basename(checkpoint['args'].darknet_cfg):
                    args_diff[k] = {"args": args.__dict__[k], "checkpoint": checkpoint['args'].__dict__[k]}
            elif k == "scale":
                if "scale" not in checkpoint["args"].__dict__:
                    if args.__dict__[k] != 32:
                        args_diff[k] = {"args": args.__dict__[k], "checkpoint": 32}
            else:
                if k in checkpoint["args"] and args.__dict__[k] != checkpoint['args'].__dict__[k]:
                    args_diff[k] = {"args": args.__dict__[k], "checkpoint": checkpoint['args'].__dict__[k]}
        if len(args_diff) > 0:
            if len(args_diff) == 1 and "img_size" in args_diff:
                ok = input("You requested an image size of %s, but trained with %s. Is this on purpose?" % (
                    args_diff["img_size"]["args"], args_diff["img_size"]["checkpoint"]))
            elif len(args_diff) == 1 and "darknet_cfg" in args_diff \
                    and ((args_diff["darknet_cfg"]["args"].endswith("darknet19_448_d2.cfg")
                          and args_diff["darknet_cfg"]["checkpoint"].endswith("darknet19_448.cfg"))
                         or (args_diff["darknet_cfg"]["args"].endswith("darknet19_448_d1.cfg")
                             and args_diff["darknet_cfg"]["checkpoint"].endswith("darknet19_448_nodilation.cfg"))):
                ok = True
            else:
                raise ValueError("We cannot reload the model from %s! Please check the args:\n%s"
                                 % (str(args.paths.pretrain_model), str(args_diff)))
    return checkpoint


def save_best_checkpoint(args, model, optimizer, scheduler, epoch, id):
    if args.keep:
        Log.debug("Save best model to %s" % args.paths.best_model)
        save_checkpoint_to(args.paths.best_model, model, optimizer, scheduler, epoch, id, args)


def save_checkpoint(args, model, optimizer, scheduler, epoch, id):
    if args.keep:
        # TODO: save model name!
        if epoch % args.checkpoint_iteration == 0:
            path = args.paths.generate_epoch_model_path(epoch)
        else:
            path = args.paths.model

        Log.debug("Save model to %s" % path)
        save_checkpoint_to(path, model, optimizer, scheduler, epoch, id, args)


def save_checkpoint_to(path, model, optimizer, scheduler, epoch, id, args):
    if args.keep:
        model_to_save = model.module if hasattr(model, "module") else model
        save_args = copy(args)
        save_args.paths = None
        state = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'ID': id,
            'args': save_args,
            'timestamp': str(datetime.now())
        }

        if scheduler:
            state['sheduler_state_dict'] = scheduler.state_dict()

        torch.save(state, path)


if __name__ == '__main__':
    from yolino.utils.general_setup import general_setup

    args = general_setup("Model Factory", config_file="params.yaml")
    get_checkpoint(args, check_args=False, print_debug=True)
