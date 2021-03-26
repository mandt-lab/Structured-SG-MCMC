import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

from lvi.optimizers import SGLD, pSGLD

from abc import ABC, abstractmethod 

class DeterministicTensor(nn.Module):

    def __init__(
        self,
        tensor_size,
        num_total_param_groups,
        init_values=None,
    ):
        super(DeterministicTensor, self).__init__()

        assert(isinstance(tensor_size, tuple) and (len(tensor_size) == 2))
        if init_values is not None:
            assert(isinstance(init_values, torch.FloatTensor))
            assert(len(init_values.shape) == len(tensor_size))
            assert(all(x==y for x,y in zip(tensor_size, init_values.shape)))

        # Create tensor of the current weights (also referred to as the 'actual' weights)
        # We want to keep track of gradients for these
        self.theta_actual = nn.Parameter(
            data=init_values if init_values is not None else torch.randn(size=tensor_size)*1e-3,
            requires_grad=True,
        )  # size: tensor_size
        self.tensor_size = tensor_size
        self.num_total_param_groups = num_total_param_groups

    def append_to_chain(self, parameter_group_idx_to_update):
        pass

    def log_prior(self, *args, **kwargs):
        return 0.0

    def sample(
        self,
        batch_size,
        parameter_group_sample_idx=None,
        parameter_group_mask=None,
        deterministic=True,
    ):
        return self.theta_actual.unsqueeze(0).expand(batch_size, *self.tensor_size)

class StochasticTensor(nn.Module):

    def __init__(
        self, 
        tensor_size, 
        param_group_ids, 
        num_total_param_groups,
        chain_length,
        prior_std,
        init_values=None, 
    ):
        super(StochasticTensor, self).__init__()

        # Check input types
        assert(isinstance(tensor_size, tuple) and (len(tensor_size) == 2))
        assert(isinstance(param_group_ids, (int, torch.LongTensor)))
        assert(isinstance(num_total_param_groups, int) and num_total_param_groups >= 1)
        assert(isinstance(chain_length, int) and chain_length >= 1)

        if init_values is not None:
            assert(isinstance(init_values, torch.FloatTensor))
            assert(len(init_values.shape) == len(tensor_size))
            assert(all(x==y for x,y in zip(tensor_size, init_values.shape)))

        # Create tensor of the current weights (also referred to as the 'actual' weights)
        # We want to keep track of gradients for these
        self.theta_actual = nn.Parameter(
            data=init_values if init_values is not None else torch.randn(size=tensor_size)*1e-3,
            requires_grad=True,
        )  # size: tensor_size

        # Create placeholder tensor that represents the empirical distribution of the values in self.theta_actual
        # No gradients should be tracked on these as we want to keep them frozen in time
        self.register_buffer(
            name="theta_chains",
            tensor=torch.zeros(size=(chain_length,) + tensor_size, dtype=torch.float32),
        )  # size: chain_length * tensor_size
        # `register_buffer` is used because we don't want to track gradients for these samples, but we 
        # would like them to be transfered to the GPU when appropriate

        # Create a tensor of the same size as theta_actual that at position (i,j,...) contains the parameter
        # group id from 0 to (num_total_param_groups-1) for theta_actual[i,j,...].
        if isinstance(param_group_ids, torch.LongTensor):
            assert(len(param_group_ids.shape) == len(tensor_size))
            assert(all(x==y for x,y in zip(param_group_ids.shape, tensor_size)))
        else:
            param_group_ids = torch.full(size=tensor_size, fill_value=param_group_ids)
        self.register_buffer(
            name="parameter_map",
            tensor=param_group_ids,
        )  # size: tensor_size

        self.num_total_param_groups = num_total_param_groups
        self.tensor_size = tensor_size
        self.chain_length = chain_length
        self.prior_dist = torch.distributions.normal.Normal(
            loc=torch.zeros_like(self.theta_actual),
            scale=prior_std,
        )

    def append_to_chain(self, parameter_group_idx_to_update):
        assert(len(parameter_group_idx_to_update.shape) == 1)
        assert(parameter_group_idx_to_update.shape[0] == self.num_total_param_groups)

        chain_idx_to_update = F.embedding(
            self.parameter_map, 
            parameter_group_idx_to_update.unsqueeze(1),
        ).permute(2, 0, 1)  # size: 1 * self.tensor_size

        self.theta_chains.scatter_(
            dim=0,
            index=chain_idx_to_update,
            src=self.theta_actual.unsqueeze(0),
        )

    def log_prior(self, parameter_groups_updated=None):
        if parameter_groups_updated is None:
            return self.prior_dist.log_prob(self.theta_actual).sum()
        else:
            return (self.prior_dist.log_prob(self.theta_actual) * \
                F.embedding(self.parameter_map, parameter_groups_updated)).sum()

    def sample(
        self,
        batch_size,
        parameter_group_sample_idx=None,
        parameter_group_mask=None,  # values of 1 indicate a group will be graded, 0 indicates a group will be sampled
        deterministic=False,
    ):
        if deterministic:
            return self.theta_actual.unsqueeze(0).expand(batch_size, *self.tensor_size)

        assert(parameter_group_sample_idx is not None)
        assert(len(parameter_group_sample_idx.shape) == 2)
        assert(parameter_group_sample_idx.shape[0] == self.num_total_param_groups)

        with torch.no_grad():
            if parameter_group_mask is None:
                parameter_group_mask = torch.ones_like(parameter_group_sample_idx, dtype=torch.float32)
            else:
                assert(len(parameter_group_mask.shape) == 2)
                assert(all(x==y for x,y in zip(parameter_group_sample_idx.shape, parameter_group_mask.shape)))

            
            # Map out which iteration in the chain to sample from for each weight.
            weight_sample_idx = F.embedding(self.parameter_map, parameter_group_sample_idx).permute(2, 0, 1)

            # Sample from chains.
            theta_samples = torch.gather(self.theta_chains, 0, weight_sample_idx)

            # Map out which of the samples need to be kept and which to be replaced with the true weight values.
            sample_mask = F.embedding(self.parameter_map, parameter_group_mask).permute(2, 0, 1)

        # Create final composite tensor.
        theta_composite = (1-sample_mask) * theta_samples + sample_mask * self.theta_actual

        return theta_composite

class StochasticNetwork(nn.Module, ABC):

    def __init__(
        self,
        tensor_dict,
        num_total_param_groups,
        chain_length,
        dropout_prob=None,
    ):
        super(StochasticNetwork, self).__init__()

        assert(isinstance(tensor_dict, dict))
        for k,v in tensor_dict.items():
            assert(isinstance(v, (StochasticTensor, DeterministicTensor)))

        self.tensor_dict = nn.ModuleDict(tensor_dict)
        self.chain_length = chain_length
        self.num_total_param_groups = num_total_param_groups
        self.use_dropout = dropout_prob is not None
        if not self.use_dropout:
            dropout_prob = 1.0  # never use samples
        self.dropout_prob = dropout_prob
        self.register_buffer(
            name="dropout_prob_tensor",
            tensor=torch.full(size=(1, num_total_param_groups), fill_value=dropout_prob),
        )

        # Keep track of how many samples are present per parameter group
        self.register_buffer(
            name="num_samples_per_group",
            tensor=torch.zeros(size=(num_total_param_groups,), dtype=torch.int64),
        )

    def initialize_optimizer(
        self, 
        update_determ=True, 
        update_stoch=True, 
        lr=1e-3, 
        sgd=False, 
        sgld=False, 
        psgld=False,
    ):
        assert(bool(sgd) + bool(sgld) + bool(psgld) == 1)  # xor
        assert(update_determ or update_stoch)

        params_to_update = []
        self.using_mcmc = sgld or psgld

        if update_stoch:
            stochastic_params = {"params": self.get_stochastic_params()}
            if self.using_mcmc:
                stochastic_params["addnoise"] = True
            params_to_update.append(stochastic_params)
        if update_determ:
            deterministic_params = {"params": self.get_deterministic_params()}
            if self.using_mcmc:
                stochastic_params["addnoise"] = False
            params_to_update.append(deterministic_params)

        if sgd:
            self.optimizer = SGD(params=params_to_update, lr=lr)
        elif sgld:
            self.optimizer = SGLD(params=params_to_update, lr=lr)
        elif psgld:
            self.optimizer = pSGLD(params=params_to_update, lr=lr)
        else:
            raise Exception("Must use either SGD, SGLD, or pSGLD optimizers.")


    @abstractmethod
    def get_stochastic_params(self):
        pass
    
    @abstractmethod
    def get_deterministic_params(self):
        pass

    def sample_weights(
        self, 
        deterministic=False,
        batch_size=None,
        for_training=False,
    ):
        if batch_size is None:
            if for_training:
                batch_size = self.num_total_param_groups  # == 1 for SGLD
            else:
                batch_size = 1

        if deterministic:
            samples = {k: v.sample(
                batch_size=batch_size, 
                deterministic=True,
            ) for k,v in self.tensor_dict.items()}

            parameter_groups_updated = torch.ones_like(self.num_samples_per_group)
            #return samples, None
        else:
            assert self.num_samples_per_group.min() > 0, "Trying to sample from chains prior to being initialized."
            parameter_group_sample_idx = torch.randint(
                size=(batch_size, self.num_total_param_groups),
                high=self.chain_length,
                low=0,
                device=self.num_samples_per_group.device,
            ).remainder(self.num_samples_per_group)  # ensure we don't sample past the current number of samples per group

            if for_training:
                if self.use_dropout:
                    # Dropout LVI allows for potentially multiple independent parameter groups to be graded simultaneously
                    parameter_group_mask = torch.bernoulli(self.dropout_prob_tensor.expand(batch_size, -1))
                else:
                    # Non-Dropout LVI will have a single parameter group graded per instance
                    parameter_group_mask = torch.eye(self.num_total_param_groups, device=self.num_samples_per_group.device)
            else:
                # During prediction time, we want to solely use samples from our distribution
                parameter_group_mask = torch.zeros_like(self.dropout_prob_tensor.expand(batch_size, -1))
    
            samples = {k: v.sample(
                batch_size=batch_size, 
                parameter_group_sample_idx=parameter_group_sample_idx.t(),
                parameter_group_mask=parameter_group_mask.t(),
            ) for k,v in self.tensor_dict.items()}
            parameter_groups_updated = (parameter_group_mask.sum(dim=0) > 0).long()

        return {
            "samples": samples, 
            "parameter_groups_updated": parameter_groups_updated,
            "vi_batch_size": batch_size,
        }

    def append_chains(self, parameter_groups_updated):
        # This method assumes that the actual weights have already been updated from the optimizer
        self.num_samples_per_group += parameter_groups_updated

        # `-1` to convert from length to current index
        # `remainder` to wrap indices >= `self.chain_length` to be between `0` and `self.chain_length-1`
        parameter_group_idx_to_update = (self.num_samples_per_group-1).remainder(self.chain_length)

        for k,v in self.tensor_dict.items():
            # For the tensor, take the current actual weights and insert them into the appropriate index
            # for their group's chain
            v.append_to_chain(
                parameter_group_idx_to_update=parameter_group_idx_to_update,
            )

    def init_chains(self):
        self.append_chains(torch.ones_like(self.num_samples_per_group))

    def get_chains(self):
        chains = {}
        num_samples = self.num_samples_per_group.min()
        for k,v in self.tensor_dict.items():
            if isinstance(v, StochasticTensor):
                chains[k] = v.theta_chains[:num_samples, ...]
        return chains

    def get_current_weights(self, accepted_types=(StochasticTensor, DeterministicTensor)):
        weights = {}
        for k,v in self.tensor_dict.items():
            if isinstance(v, accepted_types):
                chains[k] = v.theta_actual
        return chains

    def calculate_log_prior(self, parameter_groups_updated):
        return sum(v.log_prior(parameter_groups_updated=parameter_groups_updated) for k,v in self.tensor_dict.items())

    @abstractmethod
    def forward(self, X, sampled_weights):
        pass

    def sample_pred(
        self,
        X,
        deterministic,
        vi_batch_size,
        for_training,
    ):
        sample_dict = self.sample_weights(
            deterministic=deterministic,
            batch_size=vi_batch_size,
            for_training=for_training,
        )
    
        Y_hat = self.forward(
            X=X,
            sampled_weights=sample_dict,
        )

        return Y_hat, sample_dict

    @abstractmethod
    def calculate_log_likelihood(self, y, y_hat, N):
        pass

    def training_step(
        self, 
        batch, 
        N,  # total number of data points
        vi_batch_size=None, 
        deterministic_weights=False,  # if True, performs regular SGLD update, if False performs LVI update
    ):
        self.optimizer.zero_grad()

        X, Y = batch
        Y_hat, sample_dict = self.sample_pred(
            X=X,
            deterministic=deterministic_weights,
            vi_batch_size=vi_batch_size,
            for_training=True,
        )

        log_prior = self.calculate_log_prior(sample_dict["parameter_groups_updated"])
        log_likelihood = self.calculate_log_likelihood(y=Y, y_hat=Y_hat, N=N)

        if (not deterministic_weights) and self.use_dropout:
            # We need to scale the result to account for multiple parameter groups being updated per iteration
            # in the vi_batch dimension.
            # If we use a dropout prob of p with K groups for a vi_batch size of L, then we scale the individual 
            # iterations by 1 / (expected groups per iter = p * K) and then we scale the resulting sum over
            # the vi_batch by K / L.
            log_likelihood = log_likelihood / (self.dropout_prob * self.num_total_param_groups)
            log_likelihood = log_likelihood.mean(dim=0) * self.num_total_param_groups
        else:  # either regular SGD, LVI, or SGLD update is being performed, not Dropout-LVI
            log_likelihood = log_likelihood.sum(dim=0)
            
        loss = -(log_prior + log_likelihood)
        loss.backward()
        self.optimizer.step()

        if self.using_mcmc:
            self.append_chains(parameter_groups_updated=sample_dict["parameter_groups_updated"])

        return loss.item(), Y_hat, sample_dict

    def evaluate(
        self, 
        batch, 
        N,  # total number of data points
        num_samples=None,  # if None and stochastic, uses 1 sample of the weights 
        deterministic_weights=False,  # If False, will sample from weight chains to estimate negative log likelihood. If True, will use current values of weights to calculate negative log likelihood.
    ):
        
        # We want to evaluate the performance of the model during training by taking 100 samples and doing a prediction
        
        X, Y = batch
        with torch.no_grad():
            Y_hat, sample_dict = self.sample_pred(
                X=X,
                deterministic=deterministic_weights,
                vi_batch_size=num_samples,
                for_training=False,
            )

            log_likelihood = self.calculate_log_likelihood(y=Y, y_hat=Y_hat, N=N)  # size: (num_samples,)

            nll = -log_likelihood.mean(dim=0)
            #mse = F.mse_loss(Y_hat, Y.unsqueeze(0).expand(num_samples, -1, -1))

        return nll

# Regular feed forward Bayesian neural network
class FF_BNN(StochasticNetwork):
    
    def __init__(
        self,
        num_inputs=1,
        num_outputs=1,
        num_layers=2,
        hidden_sizes=[50,50],
        activation_func=nn.ReLU,
        chain_length=5000,
        group_by_layers=False,
        use_random_groups=False,
        use_permuted_groups=False,
        max_groups=None,
        dropout_prob=None,
        stochastic_biases=False,
        prior_std=1.0,
        init_values=None,
        output_distribution="normal",
        output_dist_const_params=dict(scale=1.0),
    ):
        # Check inputs
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes] * num_layers
        else:
            assert(isinstance(hidden_sizes, list))
        assert(output_distribution in ("normal", "categorical"))
        if init_values is None:
            init_values = {}

        all_sizes = [num_inputs] + hidden_sizes + [num_outputs]
        layer_shapes = [(x,y) for x,y in zip(all_sizes[:-1], all_sizes[1:])]
        self.num_transforms = len(layer_shapes)
        self.output_distribution = output_distribution
        self.output_dist_const_params = output_dist_const_params

        if group_by_layers:
            max_groups = len(layer_shapes)
        elif use_random_groups or use_permuted_groups:
            assert(max_groups is not None)
        else:
            max_groups = 1

        # Construct tensors
        tensor_dict = {}
        for i, (h_in, h_out) in enumerate(layer_shapes):
            weight_size, bias_size = (h_in, h_out), (1, h_out)
            
            tensor_dict["W_{}".format(i)] = StochasticTensor(
                tensor_size=weight_size, 
                param_group_ids=create_param_group_ids(
                    tensor_size=weight_size,
                    num_total_param_groups=max_groups,
                    layer_id=i,
                    group_by_layers=group_by_layers,
                    use_random_groups=use_random_groups,
                    use_permuted_groups=use_permuted_groups,
                ), 
                num_total_param_groups=max_groups,
                chain_length=chain_length,
                prior_std=prior_std,
                init_values=init_values.get("W_{}".format(i), None),
            )
            if stochastic_biases:
                bias_vector = StochasticTensor(
                    tensor_size=bias_size, 
                    param_group_ids=create_param_group_ids(
                        tensor_size=bias_size,
                        num_total_param_groups=max_groups,
                        layer_id=i,
                        group_by_layers=group_by_layers,
                        use_random_groups=use_random_groups,
                        use_permuted_groups=use_permuted_groups,
                    ), 
                    num_total_param_groups=max_groups,
                    chain_length=chain_length,
                    prior_std=prior_std,
                    init_values=init_values.get("b_{}".format(i), None),
                )
            else:
                bias_vector = DeterministicTensor(
                    tensor_size=bias_size,
                    num_total_param_groups=max_groups,
                    init_values=init_values.get("b_{}".format(i), None),
                )
            tensor_dict["b_{}".format(i)] = bias_vector
        
        super(FF_BNN, self).__init__(
            tensor_dict=tensor_dict,
            num_total_param_groups=max_groups,
            chain_length=chain_length,
            dropout_prob=dropout_prob,
        )

        self.act_func = activation_func()

    def forward(
        self,
        X,
        sampled_weights,
    ):
        assert(len(X.shape) == 2)
        data_batch_size = X.shape[0]

        h = X.unsqueeze(0).expand(sampled_weights["vi_batch_size"], -1, -1)
        use_activation = False  # will flip after first transformation
        for i in range(self.num_transforms):
            weight, bias = sampled_weights["samples"][f"W_{i}"], sampled_weights["samples"][f"b_{i}"]
            if use_activation:
                h = torch.bmm(self.act_func(h), weight)
            else:
                h = torch.bmm(h, weight)
                use_activation = True
            h += bias.expand(-1, h.shape[1], -1)

        return h

    def calculate_log_likelihood(self, y, y_hat, N):
        if self.output_distribution == "normal":
            dist_y_given_x = torch.distributions.normal.Normal(
                loc=y_hat,
                **self.output_dist_const_params,
            )
        elif self.output_distribution == "categorical":
            dist_y_given_x = torch.distributions.categorical.Categorical(
                logits=y_hat,
            )
        else:
            raise NotImplementedError


        log_likelihood = (dist_y_given_x.log_prob(y))
        log_likelihood = (log_likelihood.sum(dim=-1).mean(dim=-1) * N)

        return log_likelihood

    def get_stochastic_params(self):
        params = []
        for name, param in self.named_parameters():
            assert(name.split(".")[0] == "tensor_dict")
            if isinstance(self.tensor_dict[name.split(".")[1]], StochasticTensor):
                params.append(param)
        return params
    
    def get_deterministic_params(self):
        params = []
        for name, param in self.named_parameters():
            assert(name.split(".")[0] == "tensor_dict")
            if isinstance(self.tensor_dict[name.split(".")[1]], DeterministicTensor):
                params.append(param)
        return params

# Helper function for creating parameter group mappings for tensors
def create_param_group_ids(
    tensor_size,
    num_total_param_groups,
    layer_id,
    group_by_layers,
    use_random_groups,
    use_permuted_groups,
):
    if group_by_layers:
        # every layer is it's own group
        return torch.full(
            size=tensor_size, 
            fill_value=layer_id, 
            dtype=torch.int64,
        )
    elif use_random_groups:
        # every weight is uniformly randomly assigned a parameter group 
        return torch.randint(
            size=tensor_size,
            high=num_total_param_groups,
            low=0,
        )
    elif use_permuted_groups:
        # weights are assigned parameter groups in a way that ensures
        # even representation in each group
        tensor_length = 1
        for dim in tensor_size:
            tensor_length *= dim
        return torch.arange(
            start=0,
            end=tensor_length,
        ).view(*tensor_size).remainder(num_total_param_groups)
    else:
        # LVI is disabled, regular SGLD is enabled
        # Every parameter effectively belongs to the same group now
        return torch.full(
            size=tensor_size, 
            fill_value=0, 
            dtype=torch.int64,
        )