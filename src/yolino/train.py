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
import timeit
import os
import sys

import torch
from tqdm import tqdm
from yolino.model.optimizer_factory import maybe_freeze_backbone
from yolino.runner.trainer import TrainHandler
from yolino.utils.general_setup import general_setup
from yolino.utils.logger import Log


def _dist_barrier_cuda_safe(args):
    """Flush local CUDA work before collectives — bare barrier() after train/val can deadlock NCCL."""
    if not getattr(args, "distributed", False) or not torch.distributed.is_initialized():
        return
    if args.gpu:
        torch.cuda.synchronize()
    torch.distributed.barrier()


def _setup_distributed(args):
    args.world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.rank = int(os.environ.get("RANK", "0"))
    args.local_rank = int(os.environ.get("LOCAL_RANK", str(getattr(args, "gpu_id", 0))))
    args.distributed = args.world_size > 1
    args.is_main_process = args.rank == 0

    if args.distributed:
        if args.gpu:
            torch.cuda.set_device(args.local_rank)
            args.cuda = f"cuda:{args.local_rank}"
        backend = "nccl" if args.gpu else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")
        Log.warning("DDP enabled rank=%d/%d local_rank=%d backend=%s"
                    % (args.rank, args.world_size, args.local_rank, backend))
    return args


if __name__ == "__main__":
    start = timeit.default_timer()
    try:
        args = general_setup("Training")
        args = _setup_distributed(args)
        trainer = TrainHandler(args)

        if args.gpu:
            if args.gpu_id >= 0:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
                torch.cuda.set_device(args.gpu_id)
            Log.debug(
                "CUDA: available=%s current_device=%d count=%d name=%s"
                % (
                    torch.cuda.is_available(),
                    torch.cuda.current_device(),
                    torch.cuda.device_count(),
                    torch.cuda.get_device_name(0) if torch.cuda.device_count() else "n/a",
                )
            )

        if args.is_main_process:
            Log.time(key="setup", value=(timeit.default_timer() - start))
        # Last completed epoch index (for on_training_finished). If no training loop runs (e.g. resumed
        # at epoch == args.epoch), use model_epoch - 1 so `epoch` is always defined.
        last_trained_epoch = max(int(trainer.model_epoch) - 1, -1)
        if int(trainer.model_epoch) >= int(args.epoch):
            Log.error(
                "No training epochs to run: checkpoint epoch=%d but --epoch=%d (train loop is "
                "range(checkpoint_epoch, epoch), exclusive end). Increase --epoch (e.g. fine-tune smoke: "
                "--epoch %d when resuming from this checkpoint)."
                % (trainer.model_epoch, args.epoch, int(trainer.model_epoch) + 1)
            )
            sys.exit(1)
        for epoch in range(trainer.model_epoch, args.epoch):
            last_trained_epoch = epoch
            epoch_start = timeit.default_timer()

            if args.is_main_process:
                Log.debug("")
                Log.print('**** Epoch %d/%s %s ****' % (epoch, args.epoch, args.id))

            maybe_freeze_backbone(args, trainer.model, epoch=epoch)
            if getattr(trainer, "train_sampler", None) is not None:
                trainer.train_sampler.set_epoch(epoch)

            if args.distributed:
                _dist_barrier_cuda_safe(args)

            ###### TRAIN #######
            pbar = tqdm(enumerate(trainer.loader), total=len(trainer.loader), desc="Train %s" % args.id,
                        disable=not args.is_main_process)
            profile_dl = bool(getattr(args, "profile_dataloader", False))
            dl_data_wait_sum = 0.0
            dl_step_compute_sum = 0.0
            dl_profile_steps = 0
            batch_fetch_start = timeit.default_timer()

            for i, data in pbar:
                try:
                    if profile_dl:
                        data_wait = timeit.default_timer() - batch_fetch_start
                        dl_data_wait_sum += data_wait
                        dl_profile_steps += 1

                    if len(data) == 6:
                        images, grid_tensor, fileinfo, duplicate_info, params, e2e_gt_pack = data
                    else:
                        images, grid_tensor, fileinfo, duplicate_info, params = data
                        e2e_gt_pack = None
                    for j, f in enumerate(fileinfo):
                        trainer.dataset.params_per_file[f] = {}
                        for k, v in params.items():
                            trainer.dataset.params_per_file[f].update({k: v[j].item()})

                    inference_start = timeit.default_timer()

                    # 2. [중요] trainer에서 loss와 preds를 받습니다.
                    # (trainer.py에서 return sum_loss.detach().item(), outputs 로 수정했을 때 기준)
                    batch_loss, preds = trainer(fileinfo, images, grid_tensor, epoch=epoch, image_idx_in_batch=i,
                                               first_run=(i == 0), is_train=True, e2e_gt_pack=e2e_gt_pack)
                    # --- [스마트 분기 처리] 튜플이면 0번째(geom)만 쓰고, 아니면 통째로 씁니다 ---
                    if isinstance(preds, tuple):
                        eval_preds = preds[0]
                    else:
                        eval_preds = preds
                    # -----------------------------------------------------------
                    
                    # # --- [추가/수정된 부분] 튜플을 풀어서 다시 하나로 합칩니다 ---
                    # geom_preds, embed_preds = preds
                    # combined_preds = torch.cat([geom_preds, embed_preds], dim=-1)
                    # # -----------------------------------------------------------
                    
                    # 3. 실시간 Loss를 게이지 옆에 표시합니다.
                    pbar.set_postfix({'loss': f'{batch_loss:.4f}'})

                    step_compute = timeit.default_timer() - inference_start
                    Log.time(key="infer", value=step_compute)
                    if profile_dl:
                        dl_step_compute_sum += step_compute
                        batch_fetch_start = timeit.default_timer()

                    num_duplicates = int(sum(duplicate_info["total_duplicates_in_image"]).item())
                    if args.is_main_process:
                        trainer.on_images_finished(preds=eval_preds.detach().cpu(), grid_tensor=grid_tensor, epoch=epoch,
                                                   filenames=fileinfo, images=images, is_train=True,
                                                   num_duplicates=num_duplicates)

                except (Exception, BaseException) as e:
                    Log.error("Error with file %s, epoch %d, iteration %d" % (str(fileinfo), epoch, i))
                    raise e
                Log.time(key="train_batch", value=timeit.default_timer() - epoch_start)
            if profile_dl and args.is_main_process and dl_profile_steps > 0:
                mean_data_wait = dl_data_wait_sum / dl_profile_steps
                mean_step_compute = dl_step_compute_sum / dl_profile_steps
                Log.scalars(
                    tag="dataloader",
                    dict={"data_wait": mean_data_wait, "step_compute": mean_step_compute},
                    epoch=epoch,
                )
                Log.info(
                    "dataloader profile epoch %d: data_wait=%.3fs step_compute=%.3fs (n=%d)"
                    % (epoch, mean_data_wait, mean_step_compute, dl_profile_steps)
                )
            if trainer.scheduler is not None and not getattr(args, "scheduler_step_per_batch", True):
                trainer.scheduler.step()

            # DDP: rank 0 만 on_train_epoch_finished(체크포인트 등) 을 실행한다. 검증 에폭이 아닐 때 다른 rank 가
            # 이 콜백을 기다리지 않고 다음 epoch 학습으로 들어가면 첫 backward 에서 교착된다 (로그상 Train 0/N 에 멈춘 것처럼 보임).
            if args.distributed:
                _dist_barrier_cuda_safe(args)

            if args.is_main_process:
                trainer.on_train_epoch_finished(epoch, fileinfo, images, preds=eval_preds.detach(), grid_tensors=grid_tensor)
                Log.time(key="train_epoch", value=timeit.default_timer() - epoch_start)
                Log.debug("Training done epoch %d" % epoch)

            if args.distributed:
                _dist_barrier_cuda_safe(args)

            ###### EVAL #######
            # 검증은 rank 0 만 돌지만, DDP 에서 다른 rank 가 다음 epoch 으로 먼저 가면 학습 단계에서
            # 집합 연산으로 영원히 대기한다. 따라서 검증 라운드가 끝난 뒤 barrier 로 동기화하고,
            # early-stop 은 broadcast 로 모든 rank 가 같이 빠져나오게 한다.
            local_early_stop = 0
            if trainer.is_time_for_val(epoch):
                if args.is_main_process:
                    Log.debug("")
                    Log.print('**** EPOCH %d EVALUATION %s ****' % (epoch, args.id))
                    with torch.no_grad():

                        eval_batch_time = timeit.default_timer()
                        for i, data in enumerate(tqdm(trainer.val_loader, desc="Eval %s" % args.id)):
                            if len(data) == 6:
                                images, grid_tensor, fileinfo, duplicate_info, params, e2e_gt_pack = data
                            else:
                                images, grid_tensor, fileinfo, duplicate_info, params = data
                                e2e_gt_pack = None
                            for j, f in enumerate(fileinfo):
                                trainer.val_dataset.params_per_file[f] = {}
                                for k, v in params.items():
                                    trainer.val_dataset.params_per_file[f].update({k: v[j].item()})

                            _, preds = trainer(fileinfo, images, grid_tensor, epoch=epoch, image_idx_in_batch=i,
                                               is_train=False, e2e_gt_pack=e2e_gt_pack)
                            # --- [스마트 분기 처리] 튜플이면 0번째(geom)만 쓰고, 아니면 통째로 씁니다 ---
                            if isinstance(preds, tuple):
                                eval_preds = preds[0]
                            else:
                                eval_preds = preds
                            # -----------------------------------------------------------

                            num_duplicates = int(sum(duplicate_info["total_duplicates_in_image"]).item())
                            trainer.on_images_finished(preds=eval_preds.detach().cpu(), grid_tensor=grid_tensor,
                                                       epoch=epoch,
                                                       filenames=fileinfo, images=images, is_train=False,
                                                       num_duplicates=num_duplicates)

                            Log.time(key="eval_batch", value=timeit.default_timer() - eval_batch_time)
                    trainer.on_val_epoch_finished(epoch)
                    Log.time(key="eval_epoch_finished", value=timeit.default_timer() - eval_batch_time)

                    if trainer.is_converged(epoch):
                        local_early_stop = 1

                if args.distributed:
                    _dist_barrier_cuda_safe(args)
                    _dev = torch.device(args.cuda if args.gpu else "cpu")
                    _stop = torch.zeros(1, dtype=torch.long, device=_dev)
                    if local_early_stop:
                        _stop.fill_(1)
                    torch.distributed.all_reduce(_stop, op=torch.distributed.ReduceOp.MAX)
                    if int(_stop.item()) != 0:
                        break
                elif local_early_stop:
                    break

            # if epoch == 1 or epoch == args.eval_iteration:
            if args.is_main_process:
                Log.time(key="epoch", value=timeit.default_timer() - epoch_start)

        finish_start = timeit.default_timer()
        if args.is_main_process:
            trainer.on_training_finished(epoch=last_trained_epoch, do_nms=args.nms)
            Log.time(key="finish", value=timeit.default_timer() - finish_start)
        if args.distributed:
            _dist_barrier_cuda_safe(args)
            torch.distributed.destroy_process_group()
    except (Exception, BaseException) as e:
        if "args" in locals() and getattr(args, "distributed", False) and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        Log.finish()
        raise e
