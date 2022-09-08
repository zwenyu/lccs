import torch
import torch.nn as nn


class LCCS(nn.Module):
    """LCCS module that returns a version of input sample augmented by LCCS parameters.
    """

    def __init__(self, num_features):
        super().__init__()
        self.eps = 1e-5
        self.num_features = num_features
        # source BN statistics and parameters
        # named in same way as in source model so as to load statistics and parameters
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.register_buffer('running_mean', torch.empty(num_features))
        self.register_buffer('running_var', torch.empty(num_features))
        self.register_buffer('running_std', torch.empty(num_features))
        # support BN statistics
        self.register_buffer('support_mean', torch.empty(num_features))
        self.register_buffer('support_var', torch.empty(num_features))
        self.register_buffer('support_std', torch.empty(num_features))
        # combined BN statistics
        self.register_buffer('lccs_mean', torch.empty(num_features))
        self.register_buffer('lccs_var', torch.empty(num_features))
        self.register_buffer('lccs_std', torch.empty(num_features))

        self._use_stats = 'source'
        self._update_stats = 'no_update'

    def __repr__(self):
        return f'LCCS'

    def compute_source_stats(self):
        self.running_std = torch.sqrt(self.running_var + self.eps)

    def set_coeff(self, support_coeff_init, source_coeff_init):
        # set LCCS parameters as fixed scalars
        self.support_coeff_init = support_coeff_init
        self.source_coeff_init = source_coeff_init

    def set_svd_dim(self, svd_dim):
        # subtract one from dimension because first vector is support BN statistics
        self.additional_svd_dim = svd_dim - 1
        if svd_dim > 1:
            # support statistics basis vectors
            self.register_buffer('support_mean_svdbasis', torch.empty((self.num_features, self.additional_svd_dim)))
            self.register_buffer('support_std_svdbasis', torch.empty((self.num_features, self.additional_svd_dim)))
            # support LCCS parameters
            self.register_buffer('support_mean_basiscoeff', torch.empty(self.additional_svd_dim))
            self.register_buffer('support_std_basiscoeff', torch.empty(self.additional_svd_dim))

    def initialize_trainable(self, support_coeff_init, source_coeff_init):
        # initializing trainable LCCS parameters
        # source LCCS parameters
        source_mean_coeff = torch.empty(1).cuda()
        torch.nn.init.constant_(source_mean_coeff, source_coeff_init)
        self.source_mean_coeff = nn.Parameter(source_mean_coeff, requires_grad=True)
        source_std_coeff = torch.empty(1).cuda()
        torch.nn.init.constant_(source_std_coeff, source_coeff_init)
        self.source_std_coeff = nn.Parameter(source_std_coeff, requires_grad=True)

        # support LCCS parameters
        support_mean_coeff = torch.empty(1).cuda()
        torch.nn.init.constant_(support_mean_coeff, support_coeff_init)
        self.support_mean_coeff = nn.Parameter(support_mean_coeff, requires_grad=True)
        support_std_coeff = torch.empty(1).cuda()
        torch.nn.init.constant_(support_std_coeff, support_coeff_init)
        self.support_std_coeff = nn.Parameter(support_std_coeff, requires_grad=True)

        self.support_mean_basiscoeff = nn.Parameter(torch.clone(self.support_mean_basiscoeff).cuda(), requires_grad=True)
        self.support_std_basiscoeff = nn.Parameter(torch.clone(self.support_std_basiscoeff).cuda(), requires_grad=True)

    def set_use_stats_status(self, status):
        self._use_stats = status

    def set_update_stats_status(self, status):
        self._update_stats = status

    # version where support batch statistics are first spanning vectors

    def forward(self, x):
        if self._update_stats == 'initialize_support':
            # compute support BN statistics
            self.support_mean = x.mean(dim=[0, 2, 3])
            self.support_var = x.var(dim=[0, 2, 3], unbiased=False)
            self.support_std = torch.sqrt(self.support_var + self.eps) 

        elif self._update_stats == 'update_support_by_momentum':
            # update support BN statistics by momemtum
            alpha = 0.9
            self.support_mean = alpha * self.support_mean.detach() + (1 - alpha) * x.mean(dim=[0, 2, 3]).detach()
            self.support_var = alpha * self.support_var.detach() + (1 - alpha) * x.var(dim=[0, 2, 3], unbiased=False).detach()
            self.support_std = torch.sqrt(self.support_var + self.eps)

        elif self._update_stats == 'compute_support_svd':
            # compute support statistics basis vectors
            # support BN statistics
            self.support_mean = x.mean(dim=[0, 2, 3]).detach()
            self.support_var = x.var(dim=[0, 2, 3], unbiased=False).detach()
            self.support_std = torch.sqrt(self.support_var + self.eps).detach()
            unit_support_mean = (self.support_mean / torch.norm(self.support_mean, p=2)).view(-1, 1)
            unit_support_std = (self.support_std / torch.norm(self.support_std, p=2)).view(-1, 1)
            # support statistics basis vectors after subtracting away support BN statistics
            mean_matrix = x.mean(dim=[2, 3]).transpose(1, 0)
            mean_matrix -= torch.matmul(torch.matmul(unit_support_mean, unit_support_mean.T), mean_matrix)
            std_matrix = torch.sqrt(x.var(dim=[2, 3]) + self.eps).transpose(1, 0)
            std_matrix -= torch.matmul(torch.matmul(unit_support_std, unit_support_std.T), std_matrix)

            mean_u, mean_s, mean_v = torch.svd(mean_matrix)
            mean_vh = mean_v.t()
            self.support_mean_svdbasis = (mean_u @ torch.diag_embed(mean_s)[:, :self.additional_svd_dim]).detach()
            self.support_mean_basiscoeff = torch.zeros_like(mean_vh[:self.additional_svd_dim].mean(dim=[1]))

            std_u, std_s, std_v = torch.svd(std_matrix)
            std_vh = std_v.t()
            self.support_std_svdbasis = (std_u @ torch.diag_embed(mean_s)[:, :self.additional_svd_dim]).detach()
            self.support_std_basiscoeff = torch.zeros_like(std_vh[:self.additional_svd_dim].mean(dim=[1]))

        if self._use_stats == 'source':
            return ((x - self.running_mean.view(1, -1, 1, 1)) / self.running_std.view(1, -1, 1, 1)) * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

        elif self._use_stats == 'initialization_stage':
            self.lccs_mean = self.support_coeff_init * self.support_mean + self.source_coeff_init * self.running_mean
            self.lccs_std = self.support_coeff_init * self.support_std + self.source_coeff_init * self.running_std
            return ((x - self.lccs_mean.view(1, -1, 1, 1)) / self.lccs_std.view(1, -1, 1, 1)) * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

        elif self._use_stats == 'gradient_update_stage':
            self.lccs_mean = self.support_mean_coeff * self.support_mean + (self.support_mean_svdbasis @ self.support_mean_basiscoeff) + self.source_mean_coeff * self.running_mean
            self.lccs_std = self.support_std_coeff * self.support_std + (self.support_std_svdbasis @ self.support_std_basiscoeff) + self.source_std_coeff * self.running_std
            return ((x - self.lccs_mean.view(1, -1, 1, 1)) / self.lccs_std.view(1, -1, 1, 1)) * self.weight.view(1,- 1, 1, 1) + self.bias.view(1, -1, 1, 1)

        elif self._use_stats == 'evaluation_stage':
            return ((x - self.lccs_mean.detach().view(1, -1, 1, 1)) / self.lccs_std.detach().view(1, -1, 1, 1)) * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
