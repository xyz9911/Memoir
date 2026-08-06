import itertools
import os
import json
import time
import numpy as np
from collections import defaultdict

import torch
from tensorboardX import SummaryWriter

from utils.misc import set_random_seed
from utils.logger import write_to_record_file, print_progress, timeSince
from utils.distributed import init_distributed, is_default_gpu
from utils.distributed import all_gather, merge_dist_results

from utils.data import ImageFeaturesDB
from nav_src.envs.r2r.data_utils import construct_gsa_instrs, construct_instrs, get_scans
from nav_src.envs.r2r.gsa_train_env import GSATrainR2RNavBatch
from nav_src.envs.r2r.gsa_env import GSAR2RNavBatch
from nav_src.gsa_nav.parser import parse_args

from nav_src.gsa_nav.agent_gsa import GMapNavAgent


def build_dataset(args, num_robots, rank=0, is_test=False):
    feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size)

    if args.train_tour:
        train_dataset_class = GSAR2RNavBatch
    else:
        train_dataset_class = GSATrainR2RNavBatch
    eval_dataset_class = GSAR2RNavBatch

    # because we don't use distributed sampler here
    # in order to make different processes deal with different training examples
    # we need to shuffle the data with different seed in each processes
    if args.aug is not None:
        aug_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [args.aug],
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len,
            is_test=is_test, is_aug=True
        )
        aug_env = train_dataset_class(
            feat_db, aug_instr_data, args.connectivity_dir, num_robots=num_robots,
            batch_size=args.batch_size, angle_feat_size=args.angle_feat_size,
            seed=args.seed + rank, name='aug',
        )
    else:
        aug_env = None

    train_instr_data = construct_gsa_instrs(
        os.path.join(args.anno_dir, "ESA_Dataset"), args.dataset, ['train'],
        tokenizer=args.tokenizer, max_instr_len=args.max_instr_len,
        is_test=is_test
    )
    train_env = train_dataset_class(
        feat_db, train_instr_data, args.connectivity_dir, num_robots=num_robots,
        batch_size=args.batch_size,
        angle_feat_size=args.angle_feat_size, seed=args.seed + rank,
        name='train',
    )

    if args.split == 'mini':
        val_env_names = ['Validation_Residential_Basic']
    elif args.split == 'mini_test':
        val_env_names = ['Test_Residential_Basic']
    elif args.split == 'val':
        val_env_names = ['Validation_Residential_Basic', 'Validation_Non-residential_Basic', 'Validation_Non-residential_Scene']
    elif args.split == 'test':
        val_env_names = ['Test_Residential_Basic', 'Test_Non-residential_Basic', 'Test_Non-residential_Scene']
    else:
        raise Exception

    val_envs = {}
    for split in val_env_names:
        val_instr_data = construct_gsa_instrs(
            os.path.join(args.anno_dir, "ESA_Dataset"), args.dataset, [split],
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len,
            is_test=is_test
        )
        val_env = eval_dataset_class(
            feat_db, val_instr_data, args.connectivity_dir, num_robots=num_robots,
            batch_size=args.batch_size,
            angle_feat_size=args.angle_feat_size, seed=args.seed + rank,
            name=split,
        )  # evaluation using all objects
        val_envs[split] = val_env

    return train_env, val_envs, aug_env, val_env_names


def train(args, train_env, val_envs, aug_env=None, rank=-1, val_env_names=None):
    default_gpu = is_default_gpu(args)

    if default_gpu:
        with open(os.path.join(args.log_dir, 'training_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        writer = SummaryWriter(log_dir=args.log_dir)
        record_file = os.path.join(args.log_dir, 'train.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    agent_class = GMapNavAgent
    listner = agent_class(args, train_env, rank=rank)
    if args.max_traj_multiple > 0:
        env_max_traj_num = {k: v * args.max_traj_multiple for k, v in train_env.env_traj_num.items()}
    else:
        env_max_traj_num = {k: args.max_traj_num for k, v in train_env.env_traj_num.items()}
    listner.env_max_traj_num = env_max_traj_num

    # resume file
    start_iter = 0
    if args.resume_file:
        start_iter = listner.load(os.path.join(args.resume_file))
        if default_gpu:
            write_to_record_file(
                "\nLOAD the model from {}, iteration ".format(args.resume_file, start_iter),
                record_file
            )

    # first evaluation
    if args.eval_first:
        loss_str = "validation before training"
        for env_name, env in val_envs.items():
            listner.env = env
            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', extended_memory=args.extended_memory, iters=None)
            preds = listner.get_results()
            # gather distributed results
            preds = merge_dist_results(all_gather(preds))
            if default_gpu:
                score_summary, _ = env.eval_metrics(preds)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    if isinstance(val, list):
                        loss_str += f", {metric}: {[f'{x:.2f}' for x in val]}"
                    else:
                        loss_str += ', %s: %.2f' % (metric, val)
                loss_str += "\n"
        if default_gpu:
            write_to_record_file(loss_str, record_file)

    start = time.time()
    if default_gpu:
        write_to_record_file(
            '\nListener training starts, start iteration: %s' % str(start_iter), record_file
        )

    best_val = {k: {"spl": [], "sr": [], "state": []} for k in val_env_names}

    for idx in range(start_iter, start_iter + args.iters, args.log_every):
        listner.logs = defaultdict(list)
        interval = min(args.log_every, args.iters - idx)
        iter = idx + interval

        # Train for log_every interval
        if aug_env is None:
            listner.env = train_env
            listner.train(interval, feedback=args.feedback, extended_memory=args.extended_memory)  # Train interval iters
        else:
            jdx_length = len(range(interval // 2))
            for jdx in range(interval // 2):
                # Train with GT data
                listner.env = train_env
                listner.train(1, feedback=args.feedback, extended_memory=args.extended_memory)

                # Train with Augmented data
                listner.env = aug_env
                listner.train(1, feedback=args.feedback, extended_memory=args.extended_memory)

                if default_gpu:
                    print_progress(jdx, jdx_length, prefix='Progress:', suffix='Complete', bar_length=50)

        if default_gpu:
            # Log the training stats to tensorboard
            total = max(sum(listner.logs['total']), 1)          # RL: total valid actions for all examples in the batch
            length = max(len(listner.logs['critic_loss']), 1)   # RL: total (max length) in the batch
            KL_loss = sum(listner.logs['KL_loss']) / max(len(listner.logs['KL_loss']), 1)
            KL_overshoot_loss = sum(listner.logs['KL_overshoot_loss']) / max(len(listner.logs['KL_overshoot_loss']), 1)
            REACT_loss = sum(listner.logs['REACT_loss']) / max(len(listner.logs['REACT_loss']), 1)
            REVIS_loss = sum(listner.logs['REVIS_loss']) / max(len(listner.logs['REVIS_loss']), 1)
            IL_loss = sum(listner.logs['IL_loss']) / max(len(listner.logs['IL_loss']), 1)
            entropy = sum(listner.logs['entropy']) / total
            writer.add_scalar("policy_entropy", entropy, idx)
            writer.add_scalar("loss/KL_loss", KL_loss, idx)
            writer.add_scalar("loss/KL_overshoot_loss", KL_overshoot_loss, idx)
            writer.add_scalar("loss/REACT_loss", REACT_loss, idx)
            writer.add_scalar("loss/REVIS_loss", REVIS_loss, idx)
            writer.add_scalar("loss/IL_loss", IL_loss, idx)
            writer.add_scalar("total_actions", total, idx)
            writer.add_scalar("max_length", length, idx)
            write_to_record_file(
                "\ntotal_actions %d, max_length %d, entropy %.4f, IL_loss %.4f, KL_loss %.4f, KL_overshoot_loss %.4f, REACT_loss %.4f REVIS_loss %.4f" % (
                    total, length, entropy, IL_loss, KL_loss, KL_overshoot_loss, REACT_loss, REVIS_loss),
                record_file
            )

        if idx < 15000:
            continue

        # Run validation
        loss_str = "iter {}".format(iter)
        srs = defaultdict()
        spls = defaultdict()
        score_summaries = {}
        for env_name, env in val_envs.items():
            listner.env = env

            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', extended_memory=args.extended_memory, iters=None)
            preds = listner.get_results()
            preds = merge_dist_results(all_gather(preds))

            if default_gpu:
                score_summary, _ = env.eval_metrics(preds)
                srs[env_name] = score_summary['sr']
                spls[env_name] = score_summary['spl']
                score_summaries[env_name] = score_summary

                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    if isinstance(val, list):
                        loss_str += f", {metric}: {[f'{x:.2f}' for x in val]}"
                        for j, x in enumerate(val):
                            writer.add_scalar('%s_%s/%s' % (metric, j+1, env_name), x, idx)
                    else:
                        loss_str += ', %s: %.2f' % (metric, val)
                        writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], idx)

        for k, v in srs.items():
            mean_spl = spls[k]
            mean_sr = v
            # loss_str += ", %s " % k
            # loss_str += ', %s: %.2f' % ('sr', mean_sr)
            # loss_str += ', %s: %.2f' % ('spl', mean_spl)
            writer.add_scalar('%s/%s' % ('sr', k), mean_sr, idx)
            writer.add_scalar('%s/%s' % ('spl', k), mean_spl, idx)
            # select model by spl

        for k, v in srs.items():
            mean_spl = spls[k]
            mean_sr = v
            if k in best_val:
                if len(best_val[k]['spl']) < 3 or mean_sr + mean_spl >= best_val[k]['sr'][-1] + best_val[k]['spl'][-1]:
                    if len(best_val[k]['spl']) == 3:
                        best_val[k]['spl'].pop(-1)
                        best_val[k]['sr'].pop(-1)
                        best_val[k]['state'].pop(-1)
                    best_val[k]['spl'].append(mean_spl)
                    best_val[k]['sr'].append(mean_sr)
                    best_val[k]['state'].append('Iter %d %s' % (iter, loss_str))
                    best_val[k]['sr'], best_val[k]['spl'], best_val[k]['state'] = map(list, zip(*sorted(zip(best_val[k]['sr'], best_val[k]['spl'], best_val[k]['state']), key=lambda x: x[0] + x[1], reverse=True)))
                    listner.save(idx, os.path.join(args.ckpt_dir, f"iter_{iter}_{k}_sr_{mean_sr}_spl_{mean_spl}"))

                # full_results = {"metrics": score_summary, "results": full_preds}
                # json.dump(
                #     full_results,
                #     open(os.path.join(args.pred_dir, "%s_%s.json" % ("iter {}".format(iter), env_name)), 'w'), indent=4, separators=(',', ': ')
                # )

        if default_gpu:
            write_to_record_file(
                ('%s (%d %d%%) %s' % (timeSince(start), iter, float(iter) / args.iters * 100, loss_str)),
                record_file
            )
            write_to_record_file("BEST RESULT TILL NOW", record_file)
            for env_name in best_val:
                for state in best_val[env_name]['state']:
                    write_to_record_file(env_name + ' | ' + state + '\n', record_file)


def valid(args, train_env, val_envs, rank=-1):
    default_gpu = is_default_gpu(args)

    agent_class = GMapNavAgent
    agent = agent_class(args, train_env, rank=rank)

    if args.resume_file:
        print("Loaded the listener model at iter %d from %s" % (
            agent.load(args.resume_file), args.resume_file))

    if default_gpu:
        with open(os.path.join(args.log_dir, 'validation_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        record_file = os.path.join(args.log_dir, 'valid.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    score_summaries = {}
    srs = defaultdict()
    spls = defaultdict()
    tls = defaultdict()
    nes = defaultdict()
    ndtws = defaultdict()
    for env_name, env in val_envs.items():
        agent.logs = defaultdict(list)
        agent.env = env

        start_time = time.time()
        agent.test(
            use_dropout=False, feedback='argmax', extended_memory=args.extended_memory, iters=None)
        print(env_name, 'cost time: %.2fs' % (time.time() - start_time))
        preds = agent.get_results(detailed_output=args.detailed_output)
        preds = merge_dist_results(all_gather(preds))
        full_preds = agent.get_full_results()

        if default_gpu:
            score_summary, _ = env.eval_metrics(preds)
            srs[env_name] = score_summary['sr']
            spls[env_name] = score_summary['spl']
            tls[env_name] = score_summary['lengths']
            nes[env_name] = score_summary['nav_error']
            ndtws[env_name] = score_summary['nDTW']
            score_summaries[env_name] = score_summary

            loss_str = "Env name: %s" % env_name
            for metric, val in score_summary.items():
                if isinstance(val, list):
                    loss_str += f", {metric}: {[f'{x:.2f}' for x in val]}"
                else:
                    loss_str += ', %s: %.2f' % (metric, val)
            write_to_record_file(loss_str + '\n', record_file)

    for k, v in srs.items():
        with open(os.path.join(args.log_dir, f'valid_{k}_{args.resume_file.split("/")[-1]}.json'), 'w') as f:
            json.dump(score_summaries[k], f, indent=4)

def main():
    args = parse_args()
    num_robots = 1

    if args.world_size > 1:
        rank = init_distributed(args)
        torch.cuda.set_device(args.local_rank)
    else:
        rank = 0

    set_random_seed(args.seed + rank)
    train_env, val_envs, aug_env, val_env_names = build_dataset(args, num_robots, rank=rank, is_test=args.test)

    if not args.test:
        train(args, train_env, val_envs, aug_env=aug_env, rank=rank, val_env_names=val_env_names)
    else:
        valid(args, train_env, val_envs, rank=rank)


if __name__ == '__main__':
    main()
