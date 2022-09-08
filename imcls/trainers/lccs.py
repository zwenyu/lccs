import os
import os.path as osp
import json
import time
import random
import copy
import numpy as np

import torch
import torch.optim as optim
from torch.nn import functional as F

from dassl.utils import set_random_seed
from dassl.data.data_manager import build_data_loader
from dassl.data.datasets import build_dataset
from dassl.data.transforms import build_transform
from dassl.engine import TRAINER_REGISTRY
from dassl.evaluation import build_evaluator
from dassl.utils import load_checkpoint
from dassl.engine.dg import Vanilla
import trainers.lccs_utils.lccs_svd as optms

class AbstractLCCS(Vanilla):
    """Abstract class for LCCS trainer.
    """

    def __init__(self, cfg, batch_size=32, ksupport=1, init_epochs=10, grad_update_epochs=10,
        user_support_coeff_init=None, classifier_type='linear', finetune_classifier=False, svd_dim=1):
        """
        Args:
            cfg: configurations
            batch_size: batch size
            ksupport: number of support samples per class
            init_epochs: number of epochs in initialization stage
            grad_update_epochs: number of epochs in gradient update stage
            user_support_coeff_init: user-specified value for support LLCS parameter
            classifier_type: type of classifier
            finetune_classifier: updates classifier by gradient descent if True
            svd_dim: number of support statistics basis vectors
        """
        super().__init__(cfg)

        self.cfg = cfg
        self.batch_size = batch_size
        self.ksupport = ksupport
        self.init_epochs = init_epochs
        self.grad_update_epochs = grad_update_epochs
        self.user_support_coeff_init = user_support_coeff_init
        self.classifier_type = classifier_type
        self.finetune_classifier = finetune_classifier
        self.svd_dim = svd_dim
        self.eps = 1e-5

        self.evaluator = build_evaluator(cfg, lab2cname=self.dm.lab2cname)

    def load_model_nostrict(self, directory, epoch=None):
        """Non-strict loading of model state dict, since LCCS parameters added.
        """
        names = self.get_model_names()
        model_file = 'model.pth.tar-' + str(
            epoch
        ) if epoch else 'model-best.pth.tar'

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path)
                )

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint['state_dict']
            epoch = checkpoint['epoch']

            print(
                'Loading weights to {} '
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            self._models[name].load_state_dict(state_dict, strict=False)

    def get_ksupport_loaders(self):
        """Obtain support set.
        """
        torch.backends.cudnn.deterministic = True
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        # val and test loader with sample shuffling
        # follows dassl.data.data_manager
        dataset = build_dataset(self.cfg)
        tfm_train = build_transform(self.cfg, is_train=True)
        tfm_test = build_transform(self.cfg, is_train=False)

        # extract support samples
        np.random.seed(self.cfg.SEED)
        n = len(dataset.test)
        self.num_classes = dataset._num_classes
        support_idx = []
        for i in range(dataset._num_classes):
            idx_i = [j for j in range(n) if dataset.test[j]._label == i]
            support_idx += list(np.random.choice(idx_i, self.ksupport, replace=False))
        dataset.ksupport = [dataset.test[i] for i in support_idx]
        dataset.eval = [dataset.test[i] for i in range(n) if i not in support_idx]

        # support set for finetuning
        self.support_loader_train_transform = build_data_loader(
            self.cfg,
            sampler_type='RandomSampler',
            data_source=dataset.ksupport,
            batch_size=min(self.batch_size, self.ksupport*dataset._num_classes),
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=None
        )
        self.support_loader_test_transform = build_data_loader(
            self.cfg,
            sampler_type='RandomSampler',
            data_source=dataset.ksupport,
            batch_size=dataset._num_classes*self.ksupport,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=None
        )
        # evaluation set
        self.eval_loader = build_data_loader(
            self.cfg,
            sampler_type='RandomSampler',
            data_source=dataset.eval,
            batch_size=self.batch_size,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=None
        )

    def initialization_stage(self):
        """Initialization stage.

        Find initialization for source and support LCCS parameters.
        """
        self.model = self.model.to(torch.double)
        self.model.backbone.compute_source_stats()
        self.model.backbone.set_svd_dim(self.svd_dim)
        self.model.backbone.set_lccs_use_stats_status('initialization_stage')
        self.set_model_mode('eval')
        candidates_init = np.arange(0, 1.1, 0.1)

        if self.user_support_coeff_init is None:
            cross_entropy = {}
            with torch.no_grad():
                for i in candidates_init:
                    print(f'initialization of support LCCS param: {i}')
                    self.model.backbone.set_coeff(i, 1. - i)
                    set_random_seed(self.cfg.SEED)
                    cross_entropy_list = []
                    accuracy_list = []
                    # iterate through support set for init_epochs
                    len_support_loader_train_transform = len(self.support_loader_train_transform)
                    for j in range(self.init_epochs):
                        support_loader_train_transform_iter = iter(self.support_loader_train_transform)
                        for iterate in range(len_support_loader_train_transform):
                            if (j == 0) and (iterate == 0):
                                self.model.backbone.set_lccs_update_stats_status('initialize_support')
                            else:
                                self.model.backbone.set_lccs_update_stats_status('update_support_by_momentum')
                            batch = next(support_loader_train_transform_iter)
                            input, label = self.parse_batch_train(batch)
                            input, label = input.to(torch.double), label.to(torch.double)
                            output = self.model(input)

                    # evaluate on support set
                    self.model.backbone.set_lccs_update_stats_status('no_update')
                    len_support_loader_test_transform = len(self.support_loader_test_transform)
                    support_loader_train_transform_iter = iter(self.support_loader_test_transform)
                    for iterate in range(len_support_loader_test_transform):
                        batch = next(support_loader_train_transform_iter)
                        input, label = self.parse_batch_test(batch)
                        input, label = input.to(torch.double), label.to(torch.double)
                        output = self.model(input)
                        # cross-entropy, lower the better
                        ce_i = F.cross_entropy(output, label.long())
                        cross_entropy_list.append(float(ce_i))
                    # consolidate cross-entropy
                    cross_entropy[i] = np.mean(cross_entropy_list)

            ce_init = [cross_entropy[i] for i in candidates_init]
            print(f'candidate values: {candidates_init}')
            print(f'cross-entropy: {ce_init}')
            # pick candidate initalization with lowest cross entropy
            user_support_coeff_init = max([v for i, v in enumerate(candidates_init) if ce_init[i] == min(ce_init)])
            print(f'selected initialization of support LCCS param: {user_support_coeff_init}')
        else:
            user_support_coeff_init = self.user_support_coeff_init

        # iterate through support set for init_epochs to initialize model with selected initialization of LCCS parameters
        self.model.backbone.set_coeff(user_support_coeff_init, 1. - user_support_coeff_init)
        set_random_seed(self.cfg.SEED)
        with torch.no_grad():
            support_loader_train_transform_iter = iter(self.support_loader_test_transform)
            self.model.backbone.set_lccs_update_stats_status('compute_support_svd')
            batch = next(support_loader_train_transform_iter)
            input, label = self.parse_batch_train(batch)
            input, label = input.to(torch.double), label.to(torch.double)
            output = self.model(input)

        # initialize LCCS parameters as leanable
        self.model.backbone.initialize_trainable(support_coeff_init=user_support_coeff_init, source_coeff_init=1. - user_support_coeff_init)

    def gradient_update_stage(self):
        """Gradient update stage.

        Update trainable parameters.
        """
        print("### Finetuning LCCS params ###")
        self.model_optms = optms.configure_model(self.model, component='LCCS').cuda()
        params, param_names = optms.collect_params(self.model_optms, component='LCCS')
        optimizer = torch.optim.Adam(params,
            lr=1e-3,
            betas=(0.9, 0.999),
            weight_decay=0)
        self.model_optms.backbone.set_lccs_use_stats_status('gradient_update_stage')
        self.model_optms.backbone.set_lccs_update_stats_status('no_update')
        self.model_optms = self.train(self.model_optms, optimizer, self.support_loader_train_transform, self.support_loader_test_transform,
            grad_update_epochs=self.grad_update_epochs, num_classes=self.num_classes, classifier_type='linear',
            initialize_centroid=(self.classifier_type == 'mean_centroid'))

        if self.finetune_classifier:
            print("### Finetuning classifier ###")
            self.model_optms = optms.configure_model(self.model_optms, component='classifier')
            params, param_names = optms.collect_params(self.model_optms, component='classifier')
            optimizer = torch.optim.Adam(params, 
                lr=1e-3,
                betas=(0.9, 0.999),
                weight_decay=0)
            self.model_optms = self.train(self.model_optms, optimizer, self.support_loader_train_transform, self.support_loader_test_transform,
                grad_update_epochs=self.grad_update_epochs, num_classes=self.num_classes, classifier_type=self.classifier_type)

    def train(self, model_optms, optimizer, support_loader_train_transform, support_loader_test_transform, grad_update_epochs, num_classes,
        classifier_type='linear', initialize_centroid=False):
        """Model finetuning.
        """
        len_support_loader_train_transform = len(self.support_loader_train_transform)
        for epoch in range(grad_update_epochs):
            support_loader_train_transform_iter = iter(self.support_loader_train_transform)
            for iterate in range(len_support_loader_train_transform):
                batch = next(support_loader_train_transform_iter)
                input, label = self.parse_batch_test(batch)
                input, label = input.to(torch.double), label.to(torch.double)
                # order by label
                idx = np.argsort(label.cpu())
                input = input[idx]
                label = label[idx]

                if classifier_type == 'linear':
                    output = model_optms(input)
                    loss = F.cross_entropy(output, label.long())
                elif classifier_type == 'mean_centroid':
                    feat = model_optms.backbone(input)

                    uniqlabel = np.unique(label.cpu().numpy())
                    # form cluster centroids
                    newlabel = copy.deepcopy(label)
                    L = len(uniqlabel)
                    centroid_list = []
                    for i in range(L):
                        cluster_i = feat[label == uniqlabel[i]]
                        centroid_i = cluster_i.mean(dim=0)
                        centroid_list.append(centroid_i)
                        # relabel classes to remove missing classes in minibatch
                        newlabel[newlabel == uniqlabel[i]] = i
                    centroid = torch.stack(centroid_list).detach()
                    # obtain probability by cosine similarity
                    cossim = F.cosine_similarity(feat.unsqueeze(1), centroid, dim=-1)
                    # cross-entropy
                    newlabel = torch.tensor(newlabel, dtype=label.dtype).cuda() 
                    loss = F.cross_entropy(cossim, newlabel.long())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(f'Epoch {epoch} Iteration {iterate}: loss {loss.item()}')

        # save centroids at end of finetuning   
        if initialize_centroid:
            model_optms.backbone.set_lccs_use_stats_status('evaluation_stage')
            model_optms.backbone.set_lccs_update_stats_status('no_update')
            with torch.no_grad():
                cluster_dict = {i: [] for i in range(num_classes)}
                support_loader_train_transform_iter = iter(self.support_loader_test_transform)
                batch = next(support_loader_train_transform_iter)
                input, label = self.parse_batch_test(batch)
                input, label = input.to(torch.double), label.to(torch.double)
                feat = model_optms.backbone(input)

                # collect features per class
                for i in range(num_classes):
                    cluster_i = feat[label == i]
                    cluster_dict[i].append(cluster_i)

                # form cluster centroids
                centroid_list = []
                for i in range(num_classes):
                    cluster_i = cluster_dict[i]
                    centroid_i = torch.cat(cluster_i).mean(dim=0)
                    centroid_list.append(centroid_i)
                model_optms.centroid = torch.stack(centroid_list)

        return model_optms

    @torch.no_grad()
    def test(self):
        """Evaluation.
        """

        self.evaluator.reset()
        self.model_optms = self.model_optms.to(torch.float)
        self.model_optms.backbone.set_lccs_use_stats_status('evaluation_stage')
        self.model_optms.backbone.set_lccs_update_stats_status('no_update')       

        split = self.cfg.TEST.SPLIT
        print('Do evaluation on {} set'.format(split))
        data_loader = self.eval_loader

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            if self.classifier_type == 'linear':
                output = self.model_optms(input)
            elif self.classifier_type == 'mean_centroid':
                feat = self.model_optms.backbone(input) # n x C
                output = F.cosine_similarity(feat.unsqueeze(1), self.model_optms.centroid, dim=-1)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        # save results
        for k, v in results.items():
            tag = '{}/{}'.format(split, k + '_lccs')
            self.write_scalar(tag, v)

        self.save_path = os.path.join(self.output_dir, 'results.jsonl')
        with open(self.save_path, 'a') as f:
            f.write(json.dumps(results, sort_keys=True) + "\n")

# define trainers

# source classifier
@TRAINER_REGISTRY.register()
class LCCSk1n7(AbstractLCCS):
    def __init__(self, cfg):
        super().__init__(cfg, batch_size=32, ksupport=1, init_epochs=10, grad_update_epochs=10, svd_dim=7)
@TRAINER_REGISTRY.register()
class LCCSk5n35(AbstractLCCS):
    def __init__(self, cfg):
        super().__init__(cfg, batch_size=32, ksupport=5, init_epochs=10, grad_update_epochs=10, svd_dim=35)
@TRAINER_REGISTRY.register()
class LCCSk10n70(AbstractLCCS):
    def __init__(self, cfg):
        super().__init__(cfg, batch_size=32, ksupport=10, init_epochs=10, grad_update_epochs=10, svd_dim=70)

# mean centroid classifier
@TRAINER_REGISTRY.register()
class LCCSCentroidk1n7(AbstractLCCS):
    def __init__(self, cfg):
        super().__init__(cfg, batch_size=32, ksupport=1, init_epochs=10, grad_update_epochs=10, svd_dim=7, classifier_type='mean_centroid')
@TRAINER_REGISTRY.register()
class LCCSCentroidk5n35(AbstractLCCS):
    def __init__(self, cfg):
        super().__init__(cfg, batch_size=32, ksupport=5, init_epochs=10, grad_update_epochs=10, svd_dim=35, classifier_type='mean_centroid')
@TRAINER_REGISTRY.register()
class LCCSCentroidk10n70(AbstractLCCS):
    def __init__(self, cfg):
        super().__init__(cfg, batch_size=32, ksupport=10, init_epochs=10, grad_update_epochs=10, svd_dim=70, classifier_type='mean_centroid')