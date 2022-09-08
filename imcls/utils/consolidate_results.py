import os
import argparse
import numpy as np
import pandas as pd

def get_perf_gen(logs, unseen_index_list, selection_type, num_seed=1, eval_index=None):
    """Performance on unseen domain or specified domain, allows aggregating across domains.
    """

    acc_dict = {i: [] for i in range(num_seed)} # keys are seeds
    for unseen_index in unseen_index_list:  
        logs_path = os.path.join(logs, str(unseen_index))
        domain_index = unseen_index if (eval_index is None) else eval_index

        f_prefix = 'brief_test_accuracy'
        files = sorted([os.path.join(logs_path,f) for f in os.listdir(logs_path) \
            if os.path.isfile(os.path.join(logs_path,f)) and f.startswith(f_prefix)])[:num_seed]
        assert len(files) == num_seed

        # iterate through results of different seeds
        for i, f in enumerate(files):
            df = pd.read_csv(f)
            df = df[df.selection_type == selection_type]
            df = df[df.domain == domain_index]
            assert df.shape[0] == 1
            df_acc = df.acc.values[0]
            acc_dict[i].append(df_acc)

    # aggregate across domains per seed
    acc_agg = [np.mean(acc_dict[i]) for i in range(num_seed)]
    print('Acc: ', acc_agg)
    print('Mean:', np.mean(acc_agg))
    print('Standard deviation:', np.std(acc_agg))
    return(acc_agg)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Consolidate results')
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--source_model', type=str)
    parser.add_argument('--results_dir', type=str, default=None)
    # type of consolidation
    parser.add_argument('--selection_type', type=str, default="last")
    parser.add_argument('--aggregate', default=False, const=True, action='store_const',
        help='If true, return average accuracy across all target domains.')
    parser.add_argument('--num_seed', type=int, default=5) 
    # domain index for evaluation
    parser.add_argument('--eval_index', type=str, default=None,
        help='Domain evaluated; target domain if None')
    # option to run for specified test environments
    parser.add_argument('--test_env', type=str, default=None)
    args = parser.parse_args()

    if args.results_dir is None:
        args.results_dir = os.path.join(args.output_dir,
            args.dataset, args.source_model)

    if args.dataset == 'pacs':
        test_envs = ['art_painting', 'cartoon', 'photo', 'sketch']

    # consolidated performance
    print('--------- Results in {} for dataset: {} ---------'.format(
        args.results_dir, args.dataset))

    test_envs_list = test_envs if args.test_env is None else [args.test_env]

    if not args.aggregate:
        # performance for single target domain
        for d in test_envs_list:
            try:
                print("##### target domain {} #####".format(d))
                acc_d = get_perf_gen(args.results_dir, [d], args.selection_type,
                    num_seed=args.num_seed, eval_index=args.eval_index)
            except:
                print('no available result')
    else:
        # average performance across target domains
        print("##### average across target domains #####")
        acc_agg = get_perf_gen(args.results_dir, test_envs_list, args.selection_type,
            num_seed=args.num_seed, eval_index=args.eval_index)