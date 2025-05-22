import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description="Run .")

    parser.add_argument("--data_dir", nargs="?", default="./data/",
                        help="Path of data set.")
    parser.add_argument("--dataset", nargs="?", default="cora",
                        help="Dataset string.")

    parser.add_argument("--model_selec", nargs="?", default="GCN",
                        help="The model selection.")
    parser.add_argument("--fcn_out_type", nargs="?", default="stack",)
    parser.add_argument('--res', action='store_true', default=False, )

    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs. Default is 200.")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train-test split. Default is 42.")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout parameter. Default is 0.5.")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="Learning rate. Default is 0.01.")
    parser.add_argument('--wd', type=float, default=0, help='weight decay')
    parser.add_argument('--wd2', type=float, default=0,
                        help='weight decay (L2 loss on parameters).')

    parser.add_argument('--num_layer', type=int, default=2,
                        help='Number of layer.')
    # num_layer = len(layers) + 1, (input_dim, hidden1, hidden2, ..., out_dim)
    parser.add_argument('--hidden', nargs='+', type=int,
                        default=[128],
                        help='Number of hidden units.')
    parser.add_argument('--hidden_fixed', type=int, default=0,
                        help='fixed hidded dim for each layer')

    parser.add_argument('--adj_normal', type=str,
                        default='GCN_1selfLoop_2normal',)

    # MLP config
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--mlp_hidden_dim', type=int, default=0,)

    # FE
    parser.add_argument('--exp_func', type=int, default=1, help='')
    parser.add_argument('--exp_list', nargs='+', type=float,
                        default=[0.8, 0.2], help='')
    parser.add_argument("--exp_frac", type=float, default=0.7,)
    parser.add_argument('--force_decompose', action='store_true',
                        default=False, help='')
    parser.add_argument('--no_exp', action='store_true', default=False,)
    parser.add_argument('--fVer', type=str, default='v5')
    parser.add_argument('--no_frac_save', action='store_true', default=False,)

    # drop redundant
    parser.add_argument('--drop_redundant', action='store_true', default=False,)
    parser.add_argument('--drop_r_p', type=float, default=0.05,)
    parser.add_argument('--p_drop_rate', type=float, default=0.1,)
    parser.add_argument('--th_degree', type=int, default=0,)
    parser.add_argument('--th_degree_p', type=float, default=0.2,)
    
    # parser.add_argument('--use_wnh', action='store_true', default=False,)
    parser.add_argument('--th_wnh', type=float, default=1.0,)
    parser.add_argument("--wnh_ver", nargs="?", default="",)

    parser.add_argument('--alpha', type=float, default=0.2,)

    # device config
    parser.add_argument('--cuda', action='store_true', default=False,)
    parser.add_argument('--device', type=str, default="cuda",)

    # Mean Average Distance (MAD)
    parser.add_argument('--madgap', action='store_true', default=False,)
    parser.add_argument("--madgap_mask", nargs="?", default="full",)
    parser.add_argument('--val_mad', action='store_true', default=False,)

    parser.add_argument('--log_file', action='store_true', default=False,)
    parser.add_argument('--end_log', action='store_true', default=False,)
    parser.add_argument('--early_stop', action='store_true', default=False,)
    parser.add_argument('--patiences', type=int, default=50,)
    parser.add_argument('--output_skip', action='store_true', default=False,)
    parser.add_argument("--norm_type", nargs="?", default="gcn",
                        help="Adj Normalization Type")

    parser.add_argument("--jknet_mode", nargs="?", default="cat",)


    args = parser.parse_args()

    if args.dataset in ["Amazon2M", "amazon2m", "amazon_2m"]:
        args.dataset = "Amazon2M"
    elif args.dataset in ["aminer", "aminer_cs", "Aminer-CS", "Aminer_CS"]:
        args.dataset = "aminer"
    elif args.dataset in ["mag_scholar_c", "MAG_Scholar_C", "MAG_Scholar", "MAG", "mag"]:
        args.dataset = "mag_scholar_c"
    elif args.dataset in ["Reddit", "reddit"]:
        args.dataset = "reddit"
    elif args.dataset in ["Actor", "actor"]:
        args.dataset = "film"

    if args.exp_frac == 1.0 or args.exp_frac == 1:
        args.no_exp = True
        print("args.exp_frac == 1.0")
        print()

    assert args.norm_type in ["gcn"]

    return args
