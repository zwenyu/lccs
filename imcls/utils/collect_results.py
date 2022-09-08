import argparse
import json
import os

def collect_res_target(consolidate_dir):
    # find target domains
    test_env = consolidate_dir.split('/')[-1]

    # go through each seed
    subdir = [d for d in os.listdir(consolidate_dir) if os.path.isdir(os.path.join(consolidate_dir, d))]
    for grp_name in subdir:
        savename = 'brief_test_accuracy-' + grp_name + '.txt'
        save_path = os.path.join(consolidate_dir, savename)
        results_path = os.path.join(consolidate_dir, grp_name, 'results.jsonl')

        # access results file
        records = []
        try:
            with open(results_path, "r") as f:
                for line in f:
                    records.append(json.loads(line[:-1]))
        except IOError:
            print(f'IO error opening {results_path}')
            pass

        record_best = records[-1]

        # save consolidate results
        f = open(save_path, mode='w')
        f.write(','.join(['acc', 'domain', 'selection_type']))
        f.write('\n')
        acc = record_best['accuracy']
        f.write(','.join([str(acc), test_env, 'last']))
        f.write('\n')
        f.close()

def collect_res_last(consolidate_dir):
    # find target domains
    test_envs = consolidate_dir.split('/')[-1].split('-')

    # go through each seed
    subdir = [d for d in os.listdir(consolidate_dir) if os.path.isdir(os.path.join(consolidate_dir, d))]
    for grp_name in subdir:
        savename = 'brief_test_accuracy-' + grp_name + '.txt'
        save_path = os.path.join(consolidate_dir, savename)
        results_path = os.path.join(consolidate_dir, grp_name, 'results.jsonl')

        # access results file
        records = []
        try:
            with open(results_path, "r") as f:
                for line in f:
                    records.append(json.loads(line[:-1]))
        except IOError:
            print(f'IO error opening {results_path}')
            pass

        # extract entry corresponding to best model
        keys = records[0].keys()
        train_envs_key_val = [k for k in keys if ('val' in k and 'time' not in k)]

        record_best = records[-1]

        # save consolidate results
        envs = test_envs + [k.replace('val_acc_', '') for k in train_envs_key_val]
        f = open(save_path, mode='w')
        f.write(','.join(['acc', 'domain', 'selection_type']))
        f.write('\n')
        for d in envs:
            if d in test_envs:
                acc = record_best['test_acc_{}'.format(d)]
            else:
                acc = record_best['val_acc_{}'.format(d)]
            f.write(','.join([str(acc), str(d), 'last']))
            f.write('\n')
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect results')
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--source_model', type=str)
    parser.add_argument('--training_domain', type=str, choices=['source', 'target'])
    parser.add_argument('--test_env', type=str)
    parser.add_argument('--consolidate_dir', type=str, default=None)
    args = parser.parse_args()

    if args.consolidate_dir is None:
        args.consolidate_dir = os.path.join(args.output_dir,
            args.dataset, args.source_model, args.test_env)

    print('Processing algorithm in {} for dataset: {}, domain: {}...'.format(args.output_dir, args.dataset, args.test_env))

    if args.training_domain == 'source':
        collect_res_last(args.consolidate_dir)
    else:
        collect_res_target(args.consolidate_dir)

