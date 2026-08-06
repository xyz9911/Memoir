import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--root_dir', type=str, default='datasets')
    parser.add_argument('--dataset', type=str, default='r2r', choices=['r2r', 'r4r'])
    parser.add_argument('--output_dir', type=str, help='experiment id')
    parser.add_argument('--seed', type=int, default=627)

    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument("--resume_file", default="")
    parser.add_argument('--split', choices=['mini', 'val', 'test', 'mini_test'], default='mini_test')

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--train_tour', action='store_true', default=False)

    # world model
    parser.add_argument('--kl_weight', type=float, default=2.0)
    parser.add_argument('--kl_overshoot_weight', type=float, default=2.0)
    parser.add_argument('--recon_act_weight', type=float, default=1.0)
    parser.add_argument('--recon_vis_weight', type=float, default=1.0)
    parser.add_argument('--recon_act_gsnn_weight', type=float, default=1.0)
    parser.add_argument('--recon_vis_gsnn_weight', type=float, default=1.0)
    parser.add_argument('--info_nce_temperature', type=float, default=0.05)
    parser.add_argument('--info_nce_reduction', choices=['mean', 'sum'], default='mean')
    parser.add_argument('--stop_gradient', type=bool, default=False)

    parser.add_argument('--jump_friendly', action='store_true', default=False)
    parser.add_argument('--redundant_view', action='store_true', default=False)
    parser.add_argument('--share_graph', action='store_true', default=False)

    # memory
    parser.add_argument('--extended_memory', type=bool, default=True)
    parser.add_argument('--env_memory_filter', type=float, default=0.5)
    parser.add_argument('--env_memory_gamma', type=float, default=1.0)
    parser.add_argument('--min_beam_size', type=int, default=1)
    parser.add_argument('--max_beam_size', type=int, default=64)
    parser.add_argument('--env_memory_dropout', type=float, default=0.5)
    parser.add_argument('--env_memory_drop_replace', action='store_true', default=False)

    parser.add_argument('--forest', type=bool, default=False)
    parser.add_argument('--max_traj_num', type=int, default=600)
    parser.add_argument('--max_traj_multiple', type=float, default=0.0)

    parser.add_argument('--his_memory_threshold', type=float, default=0.5)
    parser.add_argument('--his_memory_gamma', type=float, default=0.5)
    parser.add_argument('--max_pairs', type=int, default=20)
    parser.add_argument('--mastermind', action='store_true', default=False)

    parser.add_argument('--exp_fusion', choices=['attention', 'similarity', 'mean'], default='similarity')
    parser.add_argument('--exp_bw', action='store_true', default=False)
    parser.add_argument('--multimodal_history', action='store_true', default=False)

    parser.add_argument('--teacher_teleport', action='store_true', default=False)
    parser.add_argument('--precise_teleport', type=bool, default=False)
    parser.add_argument('--teacher_aug', action='store_true', default=False)

    parser.add_argument('--look_ahead_steps', type=int, default=5)
    parser.add_argument('--include_neighbours', action='store_true', default=False)

    # General
    parser.add_argument('--iters', type=int, default=200000, help='training iterations')
    parser.add_argument('--log_every', type=int, default=1000)
    parser.add_argument('--eval_first', action='store_true', default=False)

    parser.add_argument('--fix_lang_embedding', action='store_true', default=False)
    parser.add_argument('--fix_pano_embedding', action='store_true', default=False)
    parser.add_argument('--fix_local_branch', action='store_true', default=False)
    parser.add_argument('--fix_pano_value', action='store_true', default=False)

    # Training Configurations
    parser.add_argument('--loss_unit', choices=['step', 'batch'], default='batch')
    parser.add_argument('--feat_dropout', type=float, default=0.3)
    parser.add_argument(
        '--optim', type=str, default='adamW',
        choices=['rms', 'adam', 'adamW', 'sgd']
    )    # rms, adam
    parser.add_argument('--lr', type=float, default=0.00001, help="the learning rate")
    parser.add_argument('--decay', dest='weight_decay', type=float, default=0.)
    parser.add_argument(
        '--feedback', type=str, default='sample',
        help='How to choose next position, one of ``teacher``, ``sample`` and ``argmax``'
    )

    parser.add_argument('--tokenizer', choices=['bert', 'xlm'], default='bert')
    parser.add_argument('--fusion', choices=['global', 'local', 'avg', 'dynamic'], default='dynamic')
    parser.add_argument('--expl_sample', action='store_true', default=False)
    parser.add_argument('--expl_max_ratio', type=float, default=0.6)
    parser.add_argument('--expert_policy', default='spl', choices=['spl', 'ndtw'])

    # distributional training (single-node, multiple-gpus)
    parser.add_argument('--world_size', type=int, default=1, help='number of gpus')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument("--node_rank", type=int, default=0, help="Id of the node")

    # Data preparation
    parser.add_argument('--max_instr_len', type=int, default=512)
    parser.add_argument('--max_action_len', type=int, default=15)
    parser.add_argument('--ignoreid', type=int, default=-100, help='ignoreid for action')

    # Augmented Paths from
    parser.add_argument("--aug", default="prevalent_aug_train")
    parser.add_argument('--bert_ckpt_file', default="datasets/R2R/pretrained/grduet_tssm_dropout.pt", help='init vlnbert')

    # Listener Model Config
    parser.add_argument("--ml_weight", type=float, default=0.20)
    parser.add_argument('--entropy_loss_weight', type=float, default=0.01)

    parser.add_argument("--img_features", type=str, default='vitbase')

    parser.add_argument('--num_l_layers', type=int, default=9)
    parser.add_argument('--num_pano_layers', type=int, default=2)
    parser.add_argument('--num_x_layers', type=int, default=4)
    parser.add_argument('--graph_sprels', action='store_true', default=True)

    # Submision configuration
    parser.add_argument("--submit", action='store_true', default=False)
    parser.add_argument('--no_backtrack', action='store_true', default=False)
    parser.add_argument('--detailed_output', action='store_true', default=False)

    # Model hyper params:
    parser.add_argument("--angle_feat_size", type=int, default=4)
    parser.add_argument('--image_feat_size', type=int, default=512)
    parser.add_argument('--obj_feat_size', type=int, default=0)
    parser.add_argument('--views', type=int, default=36)

    # # A2C
    parser.add_argument(
        "--normalize", dest="normalize_loss", default="total",
        type=str, help='batch or total'
    )
    parser.add_argument('--train_alg',
        choices=['imitation', 'dagger'],
        default='dagger'
    )

    args, _ = parser.parse_known_args()

    args = postprocess_args(args)

    return args


def postprocess_args(args):
    ROOTDIR = args.root_dir

    # Setup input paths
    ft_file_map = {
        # 'vitbase': 'pth_vit_base_patch16_224_imagenet.hdf5',
        'vitbase': 'clip_vit-b16_mp3d_hm3d_gibson.hdf5',
        'bert': 'pth_bert_base_uncased.hdf5',
    }
    args.img_ft_file = os.path.join(ROOTDIR, 'R2R', 'features', ft_file_map[args.img_features])
    # args.sent_ft_file = os.path.join(ROOTDIR, 'R2R', 'features', ft_file_map[args.sent_features])

    args.connectivity_dir = os.path.join(ROOTDIR, 'R2R', 'connectivity_full')
    # args.scan_data_dir = os.path.join(ROOTDIR, 'Matterport3D', 'v1_unzip_scans')

    args.anno_dir = os.path.join(ROOTDIR, 'R2R', 'annotations')

    # Build paths
    args.ckpt_dir = os.path.join(args.output_dir, 'ckpts')
    args.log_dir = os.path.join(args.output_dir, 'logs')
    args.pred_dir = os.path.join(args.output_dir, 'preds')

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.pred_dir, exist_ok=True)

    return args

