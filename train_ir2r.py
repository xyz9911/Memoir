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
from nav_src.envs.r2r.data_utils import construct_instrs, get_tours
from nav_src.envs.r2r.iter_env import IR2RNavBatch
from nav_src.envs.eval_utils import compute_tour_ndtw
from nav_src.iter_nav.parser import parse_args

from nav_src.iter_nav.agent_ir2r import GMapNavAgent


def build_dataset(args, num_robots, rank=0, is_test=False):
    feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size)

    dataset_class = IR2RNavBatch

    # because we don't use distributed sampler here
    # in order to make different processes deal with different training examples
    # we need to shuffle the data with different seed in each processes
    if args.aug is not None:
        aug_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [args.aug],
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len,
            is_test=is_test, is_aug=True
        )
        aug_tour_data = get_tours('aug')
        aug_env = dataset_class(
            feat_db, aug_instr_data, args.connectivity_dir, aug_tour_data, num_robots=num_robots,
            batch_size=args.batch_size, angle_feat_size=args.angle_feat_size,
            seed=args.seed + rank, name='aug',
        )
    else:
        aug_env = None

    train_instr_data = construct_instrs(
        args.anno_dir, args.dataset, ['train'],
        tokenizer=args.tokenizer, max_instr_len=args.max_instr_len,
        is_test=is_test
    )
    train_tour_data = get_tours('train')
    train_env = dataset_class(
        feat_db, train_instr_data, args.connectivity_dir, train_tour_data, num_robots=num_robots,
        batch_size=args.batch_size,
        angle_feat_size=args.angle_feat_size, seed=args.seed + rank,
        name='train',
    )

    val_env_names = ['val_seen', 'val_unseen']

    val_envs = {}
    for split in val_env_names:
        val_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [split],
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len,
            is_test=is_test
        )
        val_tour_data = get_tours(split)
        val_env = dataset_class(
            feat_db, val_instr_data, args.connectivity_dir, val_tour_data, num_robots=num_robots,
            batch_size=args.batch_size,
            angle_feat_size=args.angle_feat_size, seed=args.seed + rank,
            name=split,
        )  # evaluation using all objects
        val_envs[split] = val_env

    return train_env, val_envs, aug_env


def train(args, train_env, val_envs, aug_env=None, rank=-1):
    default_gpu = is_default_gpu(args)

    if default_gpu:
        with open(os.path.join(args.log_dir, 'training_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        writer = SummaryWriter(log_dir=args.log_dir)
        record_file = os.path.join(args.log_dir, 'train.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    agent_class = GMapNavAgent
    listner = agent_class(args, train_env, rank=rank)

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
        if default_gpu:
            write_to_record_file(loss_str, record_file)

    start = time.time()
    if default_gpu:
        write_to_record_file(
            '\nListener training starts, start iteration: %s' % str(start_iter), record_file
        )

    best_val = {'val_unseen': {"spl": 0., "sr": 0., "state": ""}}
    if args.dataset == 'r4r':
        best_val = {'val_unseen_sampled': {"spl": 0., "sr": 0., "state": ""}}

    gmaps, aug_gmaps = None, None
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
                gmaps = listner.gmaps
                if aug_gmaps is not None:
                    listner.gmaps = aug_gmaps

                # Train with Augmented data
                listner.env = aug_env
                listner.train(1, feedback=args.feedback, extended_memory=args.extended_memory)
                aug_gmaps = listner.gmaps
                if gmaps is not None:
                    listner.gmaps = gmaps

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

        # Run validation
        loss_str = "iter {}".format(iter)
        for env_name, env in val_envs.items():
            def dist_func(x, y):
                return env.hash_distance_map[x[0] * 100000 + y[0]]
            listner.env = env

            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', extended_memory=args.extended_memory, iters=None)
            preds = listner.get_results()
            preds = merge_dist_results(all_gather(preds))
            full_preds = listner.get_full_results()
            full_preds = merge_dist_results(all_gather(full_preds))

            if default_gpu:
                score_summary, all_metrics = env.eval_metrics(preds)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    if isinstance(val, list):
                        loss_str += f", {metric}: {[f'{x:.2f}' for x in val]}"
                        for j, x in enumerate(val):
                            writer.add_scalar('%s_%s/%s' % (metric, j+1, env_name), x, idx)
                    else:
                        loss_str += ', %s: %.2f' % (metric, val)
                        writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], idx)

                # process tour
                all_tours = env.tour_data
                all_tours = list(itertools.chain.from_iterable(all_tours.values()))
                all_tour_length = sum([len(tour) for tour in all_tours])
                all_metric_dict = {}
                for count, id in enumerate(all_metrics['instr_id']):
                    all_metric_dict[id] = {
                        metric_name: metric_values[count]
                        for metric_name, metric_values in all_metrics.items()
                    }
                all_metrics = all_metric_dict

                # compute tour metrics
                sr, spl, ndtw = 0, 0, 0
                for tour in all_tours:
                    weight = len(tour) / all_tour_length
                    inner_sr_list, inner_spl_list, inner_ndtw_list = [], [], []
                    inner_weight_list = []
                    edited_pred_path = []
                    edited_gt_path = []
                    for i, instr_id in enumerate(tour):
                        metrics = all_metrics[instr_id]
                        scan = metrics['scan']
                        edited_pred_path += [{"position": env.hash_node_map[scan + "_" + node[0]], "episode_id": i}
                                             for node in metrics['agent_path']]
                        edited_gt_path += [{"position": env.hash_node_map[scan + "_" + node], "episode_id": i}
                                           for node in metrics['gt_path']]
                        inner_weight_list.append(
                            max(metrics['gt_length'], metrics['trajectory_lengths']) / metrics['gt_length'])
                        inner_sr_list.append(metrics['success'])
                        inner_spl_list.append(metrics['spl'])
                        # inner_ndtw_list.append(metrics['nDTW'])
                    inner_weight_list = np.array(inner_weight_list) / sum(inner_weight_list)
                    sr += weight * (inner_weight_list * np.array(inner_sr_list)).sum()
                    spl += weight * (inner_weight_list * np.array(inner_spl_list)).sum()
                    tour_ndtw = compute_tour_ndtw(edited_pred_path, edited_gt_path, dist_func)
                    ndtw += weight * tour_ndtw
                loss_str += ', %s: %.2f' % ('t-sr', sr * 100)
                loss_str += ', %s: %.2f' % ('t-spl', spl * 100)
                loss_str += ', %s: %.2f' % ('t-nDTW', ndtw * 100)

                score_summary['t-sr'] = sr * 100
                score_summary['t-spl'] = spl * 100
                score_summary['t-nDTW'] = ndtw * 100

                # select model by spl
                if env_name in best_val:
                    if score_summary['sr'] + score_summary['spl'] >= best_val[env_name]['sr'] + best_val[env_name]['spl']:
                        best_val[env_name]['spl'] = score_summary['spl']
                        best_val[env_name]['sr'] = score_summary['sr']
                        best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                        listner.save(idx, os.path.join(args.ckpt_dir, "best_%s" % (env_name)))

                # full_results = {"metrics": score_summary, "results": full_preds}
                # json.dump(
                #     full_results,
                #     open(os.path.join(args.pred_dir, "%s_%s.json" % ("iter {}".format(iter), env_name)), 'w'), indent=4, separators=(',', ': ')
                # )

        if default_gpu:
            listner.save(idx, os.path.join(args.ckpt_dir, "latest_dict"))

            write_to_record_file(
                ('%s (%d %d%%) %s' % (timeSince(start), iter, float(iter) / args.iters * 100, loss_str)),
                record_file
            )
            write_to_record_file("BEST RESULT TILL NOW", record_file)
            for env_name in best_val:
                write_to_record_file(env_name + ' | ' + best_val[env_name]['state'] + '\n', record_file)


def valid(args, train_env, val_envs, rank=-1):
    default_gpu = is_default_gpu(args)

    agent_class = GMapNavAgent
    agent = agent_class(args, train_env, rank=rank)

    if args.resume_file is not None:
        print("Loaded the listener model at iter %d from %s" % (
            agent.load(args.resume_file), args.resume_file))

    if default_gpu:
        with open(os.path.join(args.log_dir, 'validation_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        record_file = os.path.join(args.log_dir, 'valid.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    for env_name, env in val_envs.items():
        def dist_func(x, y):
            return env.hash_distance_map[x[0] * 100000 + y[0]]

        prefix = 'submit' if args.detailed_output is False else 'detail'
        if os.path.exists(os.path.join(args.pred_dir, "%s_%s.json" % (prefix, env_name))):
            continue
        agent.logs = defaultdict(list)
        agent.env = env

        start_time = time.time()
        agent.test(
            use_dropout=False, feedback='argmax', extended_memory=args.extended_memory, iters=None)
        print(env_name, 'cost time: %.2fs' % (time.time() - start_time))
        preds = agent.get_results(detailed_output=args.detailed_output)
        preds = merge_dist_results(all_gather(preds))
        full_preds = agent.get_full_results()
        full_preds = merge_dist_results(all_gather(full_preds))

        if default_gpu:
            if 'test' not in env_name:
                score_summary, all_metrics = env.eval_metrics(preds)
                loss_str = "Env name: %s" % env_name
                for metric, val in score_summary.items():
                    if isinstance(val, list):
                        loss_str += f", {metric}: {[f'{x:.2f}' for x in val]}"
                    else:
                        loss_str += ', %s: %.2f' % (metric, val)

                # process tour
                all_tours = env.tour_data
                all_tours = list(itertools.chain.from_iterable(all_tours.values()))
                all_tour_length = sum([len(tour) for tour in all_tours])

                # transform to tour metrics
                all_metric_dict = {}
                for count, id in enumerate(all_metrics['instr_id']):
                    all_metric_dict[id] = {
                        metric_name: metric_values[count]
                        for metric_name, metric_values in all_metrics.items()
                    }
                all_metrics = all_metric_dict

                # compute tour metrics
                sr, spl, ndtw = 0, 0, 0
                for tour in all_tours:
                    weight = len(tour) / all_tour_length
                    inner_sr_list, inner_spl_list, inner_ndtw_list = [], [], []
                    inner_weight_list = []
                    edited_pred_path = []
                    edited_gt_path = []
                    for i, instr_id in enumerate(tour):
                        metrics = all_metrics[instr_id]
                        scan = metrics['scan']
                        edited_pred_path += [{"position": env.hash_node_map[scan + "_" + node[0]], "episode_id": i}
                                             for node in metrics['agent_path']]
                        edited_gt_path += [{"position": env.hash_node_map[scan + "_" + node], "episode_id": i}
                                           for node in metrics['gt_path']]
                        inner_weight_list.append(
                            max(metrics['gt_length'], metrics['trajectory_lengths']) / metrics['gt_length'])
                        inner_sr_list.append(metrics['success'])
                        inner_spl_list.append(metrics['spl'])
                        # inner_ndtw_list.append(metrics['nDTW'])
                    inner_weight_list = np.array(inner_weight_list) / sum(inner_weight_list)
                    sr += weight * (inner_weight_list * np.array(inner_sr_list)).sum()
                    spl += weight * (inner_weight_list * np.array(inner_spl_list)).sum()
                    tour_ndtw = compute_tour_ndtw(edited_pred_path, edited_gt_path, dist_func)
                    ndtw += weight * tour_ndtw
                loss_str += ', %s: %.2f' % ('t-sr', sr * 100)
                loss_str += ', %s: %.2f' % ('t-spl', spl * 100)
                loss_str += ', %s: %.2f' % ('t-nDTW', ndtw * 100)

                score_summary['t-sr'] = sr * 100
                score_summary['t-spl'] = spl * 100
                score_summary['t-nDTW'] = ndtw * 100

                write_to_record_file(loss_str + '\n', record_file)

                full_results = {"metrics": score_summary, "results": full_preds}
                json.dump(
                    full_results,
                    open(os.path.join(args.pred_dir, "%s.json" % env_name), 'w'), indent=4, separators=(',', ': ')
                )


            if args.submit:
                json.dump(
                    preds,
                    open(os.path.join(args.pred_dir, "%s_%s.json" % (prefix, env_name)), 'w'),
                    sort_keys=True, indent=4, separators=(',', ': ')
                )


def main():
    args = parse_args()
    num_robots = 1

    if args.world_size > 1:
        rank = init_distributed(args)
        torch.cuda.set_device(args.local_rank)
    else:
        rank = 0

    set_random_seed(args.seed + rank)
    train_env, val_envs, aug_env = build_dataset(args, num_robots, rank=rank, is_test=args.test)

    if not args.test:
        train(args, train_env, val_envs, aug_env=aug_env, rank=rank)
    else:
        valid(args, train_env, val_envs, rank=rank)


if __name__ == '__main__':
    main()
