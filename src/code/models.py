import logging
import safetensors
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig, GPTNeoXConfig, GPT2Config, AutoModelForCausalLM, AutoModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

########################################
# Transformer setup
########################################


def init_transformer(config):
    """
    Initializes a Transformer model with a custom configuration.

    Returns:
        model: The initialized Transformer model.
        model_type (str): The type of the model.
    """
    # set seed for reproducibility
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if config.model_type == "llama":
        model_config = LlamaConfig(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            attention_dropout=config.attention_dropout,
            use_cache=False,
        )
    elif config.model_type == "gpt-neo-x":
        model_config = GPTNeoXConfig(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            attention_dropout=config.attention_dropout,
            use_parallel_residual=False,
            attention_bias=False,
            use_cache=False,
            is_decoder=True,
            classifier_dropout=0.0,
        )
    elif config.model_type == "gpt-2":
        model_config = GPT2Config(
            n_positions=config.max_position_embeddings,
            n_head=config.num_attention_heads,
            n_embd=config.hidden_size,
            n_layer=config.num_hidden_layers,
            n_inner=config.intermediate_size,
            resid_dropout=0.0,
            embed_pdrop=0.0,
            attn_pdrop=0.0,
            activation_function="gelu",
        )
    else:
        raise ValueError(f"Model type {config.model_type} not supported")
    # Initialize the model with the custom configuration
    if config.setting == "categorical-sequence":
        # Initialize tokenizer using the centralized configuration
        model_config.bos_token_id = config.tokenizer_vocab["s"]  # Start token
        model_config.vocab_size = len(
            config.tokenizer_vocab
        )  # Length of custom tokenizer's vocab
        model = AutoModelForCausalLM.from_config(model_config)
        model.output_key = "logits"
        model.labels_required_for_inference = False

    elif config.setting == "linear-regression":
        model = LinearRegressionTransformer(config=config, model_config=model_config)

    elif config.setting == "classification":
        model = ClassificationTransformer(config=config, model_config=model_config)

    logger.info(f"Initialized model of type: {config.model_type}")
    logger.info(f"Number of parameters: {model.num_parameters()}")

    return model

def load_transformer(config, model_path):
    if config.setting == "linear-regression" or config.setting == "classification":
        model = init_transformer(config)
        model.load_pretrained(model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.output_key = "logits"
        model.labels_required_for_inference = False
    return model

# generic class for input-output ICL transformer models

class GenericICLTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def device(self):
        return next(self.parameters()).device

    def num_parameters(self):
        return (
            self._backbone.num_parameters()
            + sum(p.numel() for p in self._read_in.parameters())
            + sum(p.numel() for p in self._read_out.parameters())
        )

    def save_pretrained(self, path):
        torch.save(self.state_dict(), path)

    def load_pretrained(self, path):
        if path.endswith(".safetensors"):
            self.load_state_dict(safetensors.torch.load_file(path))
        else:
            self.load_state_dict(torch.load(path))

# Transformer model for linear regression task

class LinearRegressionTransformer(GenericICLTransformer):
    def __init__(self, config, model_config):
        super().__init__()
        self._read_in = nn.Linear(config.num_dims, config.hidden_size)
        self._backbone = AutoModel.from_config(model_config)
        self._read_out = nn.Linear(config.hidden_size, 1)
        self.output_key = "predictions"
        self.labels_required_for_inference = True

    @staticmethod
    def _combine(xs, ys):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs.shape
        ys_b_wide = torch.cat(
            (
                ys.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs, labels):
        """
        Forward method for regression.

        Args:
            interleaved_input: Interleaved input features (batch_size, seq_len, n_dims).
            labels: Input targets (batch_size, seq_len/2).

        Returns:
            A dictionary with 'predictions' (predicted values) and optionally 'loss'.
        """
        interleaved_input = self._combine(xs, labels)

        # Generate embeddings
        embeds = self._read_in(interleaved_input)

        # Pass through the transformer backbone
        output = self._backbone(inputs_embeds=embeds).last_hidden_state

        # Compute predictions for xs positions
        predictions = self._read_out(output)[
            :, ::2, 0
        ]  # Extract predictions for xs positions, shape: (batch_size, seq_len/2)

        # Prepare output dictionary
        result = {"predictions": predictions}
        # Compute mean squared error loss
        loss_fn = nn.MSELoss()
        loss = loss_fn(predictions, labels)
        result["loss"] = loss

        return result


# Transformer model for classification task
class ClassificationTransformer(GenericICLTransformer):
    def __init__(self, config, model_config):
        super().__init__()
        self._read_in = nn.Linear(config.num_dims + 1, config.hidden_size)
        self._backbone = AutoModel.from_config(model_config)
        self._read_out = nn.Linear(config.hidden_size, config.num_labels)
        self.output_key = "logits"
        self.labels_required_for_inference = False

    def forward(self, pairs, labels=None):
        """
        Forward method for classification.

        Args:
            pairs: item-label pairs (batch_size, seq_len, n_dims).
            labels: Input targets (batch_size, seq_len).

        Returns:
            A dictionary with 'logits' (predicted values) and optionally 'loss'.
        """
        # Generate embeddings
        embeds = self._read_in(pairs)

        # Pass through the transformer backbone
        output = self._backbone(inputs_embeds=embeds).last_hidden_state

        # Compute logits
        logits = self._read_out(output)[:, -1, :]  # take only logits for the last pair

        # Prepare output dictionary
        result = {"logits": logits}
        if labels is not None:  # if labels are provided, compute loss
            # Compute cross entropy loss
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            result["loss"] = loss

        return result

########################################
# Algorithmic solutions to tasks
########################################


def get_algorithmic_solutions(config, include_optimal_constant_solution=False):
    """
    Returns a list of algorithmic solutions for a given setting.
    """
    if config.setting == "categorical-sequence":
        algorithmic_solutions = {
            "memorized": BayesianAveragingCategoricalSequence(
                prior=np.load(config.train_datapath)["tasks"]
            ),
            "generalized": PosteriorMeanCategoricalSequence(
                prior=config.prior_params
            ),
        }
        if include_optimal_constant_solution:
            algorithmic_solutions["optimal_constant"] = OptimalConstantCategoricalSequence(
                prior=config.prior_params
            )
    elif config.setting == "classification":
        algorithmic_solutions = {
            "memorized": LabelMemorizingClassification(prior=np.load(config.train_datapath)["tasks"], within_class_variance=config.within_class_variance, num_labels=config.num_labels),
            "generalized": CopyingClassification(prior=config.prior_params, within_class_variance=config.within_class_variance, num_labels=config.num_labels),
        }
        if include_optimal_constant_solution:
            algorithmic_solutions["optimal_constant"] = OptimalConstantClassification(prior=config.prior_params, within_class_variance=config.within_class_variance, num_labels=config.num_labels)
    elif config.setting == "linear-regression":
        algorithmic_solutions = {
            "memorized": dMMSERegression(prior=np.load(config.train_datapath)["tasks"], noise_variance=config.noise_variance),
            "generalized": RidgeRegression(prior=config.prior_params, noise_variance=config.noise_variance),
        }
        if include_optimal_constant_solution:
            algorithmic_solutions["optimal_constant"] = OptimalConstantLinearRegression(prior=config.prior_params, noise_variance=config.noise_variance)
    else:
        raise ValueError(f"Setting {config.setting} not supported")
    return algorithmic_solutions


# Solutions to categorical sequence task

class BayesianAveragingCategoricalSequence:
    def __init__(self, prior):
        self.output_key = "logits"
        self.prior = prior
        self.labels_required_for_inference = False

    @property
    def device(self):
        # return gpu if available, otherwise cpu
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, input_ids):
        """
        Computes the next-token probabilities for each token in the sequence across multiple tasks.

        Args:
            input_ids (torch.Tensor): Tensor of shape (batch_size, context_length) containing the input ids.

        Returns:
            posterior_predictive (torch.Tensor): Tensor of shape (batch_size, context_length + 1, num_dims)
                containing the probabilities for each label to be the next in the sequence.
        """
        device = input_ids.device

        theta_matrix = torch.as_tensor(
            self.prior, device=device
        )  # Shape: (num_tasks, num_dims)
        log_theta_matrix = torch.log(theta_matrix)  # convert to log to prevent overflow

        # Remove start token
        # Assuming that input_ids[ :, 0] is start_token
        input_ids = input_ids[:, 1:]  # Shape: (batch_size, context_length)

        # Number of tasks and variables
        num_tasks, num_dims = log_theta_matrix.shape
        batch_size, context_length = input_ids.shape

        # Expand theta_matrix for broadcasting
        log_theta_expanded = log_theta_matrix.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, num_tasks, num_dims)

        # Create one-hot encoding for the next tokens
        # tokens_one_hot: (batch_size, context_length, 1, num_dims)
        tokens_one_hot = (
            torch.nn.functional.one_hot(input_ids, num_classes=num_dims)
            .float()
            .unsqueeze(2)
        )

        # cumulative sum over the one hot vector to get cumulative counts
        counts = tokens_one_hot.cumsum(
            dim=1
        )  # (batch_size, context_length, 1, num_dims)

        # broadcast log_theta_expanded and counts, and sum over num_dims to get likelihood that each task generated the sequence
        llk = (counts * log_theta_expanded).sum(
            -1
        )  # (batch_size, context_length, num_tasks)

        # assume uniform prior over theta sets 
        posterior = llk + torch.log(
            torch.tensor(1 / num_tasks).to(device)
        )  # (batch_size, context_length, num_tasks)

        # add posterior to log_theta_expanded and compute proper weighted average
        # Convert posterior to probabilities and compute weighted average
        posterior_probs = torch.softmax(posterior, dim=-1)  # normalize posterior
        posterior_predictive = (
            posterior_probs.unsqueeze(-1) * theta_matrix.unsqueeze(0).unsqueeze(0)
        ).sum(dim=-2)  # (batch_size, context_length, num_dims)
        log_posterior_predictive = torch.log(posterior_predictive)

        # insert probabilities in first position based on theta_matrix average
        initial_probs = torch.log(theta_matrix.mean(dim=0))  # (num_dims,) 
        initial_probs = initial_probs[None, None, :].expand(
            batch_size, 1, num_dims
        )  # Expand to (batch_size, 1, num_dims)

        log_posterior_predictive_full = torch.cat(
            [initial_probs, log_posterior_predictive], dim=1
        )  # Concatenate along dim=1

        # detach if on gpu
        if device.type == "cuda":
            log_posterior_predictive_full = log_posterior_predictive_full.detach().cpu()

        return {self.output_key: log_posterior_predictive_full}


class PosteriorMeanCategoricalSequence:
    def __init__(self, prior):
        self.output_key = "logits"
        self.prior = prior
        self.labels_required_for_inference = False

    @property
    def device(self):
        # return gpu if available, otherwise cpu
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, input_ids):
        """
        Computes the posterior mean of the next-token probabilities for each token in the sequence,
        incorporating a Dirichlet prior.

        Args:
            input_ids (torch.Tensor): Tensor of shape (batch_size, context_length) containing the input ids.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, context_length + 1, num_dims) containing the posterior mean probabilities.
        """
        input_ids = input_ids[:, 1:]  # Remove start token
        device = input_ids.device  # Get the device from input_ids tensor

        # Ensure alphas is a torch tensor on the correct device
        alphas = torch.as_tensor(self.prior, device=device)  # Shape: (num_dims,)
        num_dims = alphas.shape[0]
        batch_size, context_length = input_ids.shape

        # One-hot encode input_ids
        one_hot_sequences = torch.nn.functional.one_hot(
            input_ids, num_classes=num_dims
        ).float()
        # Shape: (batch_size, context_length, num_dims)

        # Compute cumulative counts
        cumsum = torch.cumsum(
            one_hot_sequences, dim=1
        )  # Shape: (batch_size, context_length, num_dims)

        # Add alphas to cumulative counts to get posterior counts
        alphas_expanded = alphas.view(1, 1, num_dims)  # Shape: (1, 1, num_dims)
        posterior_counts = (
            cumsum + alphas_expanded
        )  # Broadcasting over batch and seq_length

        # Compute total counts at each time step (sum over categories)
        total_counts = posterior_counts.sum(
            dim=-1, keepdim=True
        )  # Shape: (batch_size, context_length, 1)

        # Compute posterior mean (posterior_counts / total_counts)
        posterior_mean = (
            posterior_counts / total_counts
        )  # Shape: (batch_size, context_length, num_dims)

        # Compute initial probabilities (prior predictive) using alphas
        total_alphas = alphas.sum()  # Scalar
        initial_probs = alphas / total_alphas  # Shape: (num_dims,)
        initial_probs = initial_probs.view(1, 1, num_dims).expand(
            batch_size, -1, -1
        )  # Shape: (batch_size, 1, num_dims)

        # Concatenate initial probabilities with posterior means
        posterior_mean_probs = torch.cat(
            [initial_probs, posterior_mean], dim=1
        )  # Shape: (batch_size, context_length + 1, num_dims)

        log_posterior_mean_probs = torch.log(posterior_mean_probs)

        # detach if on gpu
        if device.type == "cuda":
            log_posterior_mean_probs = log_posterior_mean_probs.detach().cpu()

        return {self.output_key: log_posterior_mean_probs}


class OptimalConstantCategoricalSequence:
    def __init__(self, prior):
        self.output_key = "logits"
        self.prior = prior
        self.labels_required_for_inference = False

    @property
    def device(self):
        # return gpu if available, otherwise cpu
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, input_ids):
        prior_tensor = torch.as_tensor(self.prior, device=self.device)
        # duplicate the prior tensor for each element in each sequence in the batch
        prior_tensor = prior_tensor.unsqueeze(0).unsqueeze(0).expand(input_ids.shape[0], input_ids.shape[1], -1)
        # compute logits for each element in each sequence in the batch
        logits = torch.log(prior_tensor)
        return {self.output_key: logits}


# Solutions to classification task

class LabelMemorizingClassification:
    def __init__(self, prior, within_class_variance, num_labels):
        self.output_key = "logits"
        self.prior = prior  # has shape (num_tasks, context_length, n_dims+1)
        self.within_class_variance = within_class_variance
        self.labels_required_for_inference = False
        self.num_labels = num_labels

    @property
    def device(self):
        # return gpu if available, otherwise cpu
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, pairs):
        """
        Returns the label for the target task, with respect to the similarity of all other tasks in the prior distribution.

        Args:
            pairs: item-label pairs tensor of shape (batch_size, context_length, n_dims+1).

        Returns:
            A dictionary with 'logits' (logits for the target task).
        """
        device = pairs.device
        # The target item (features only)
        target_items = pairs[:, -1, :-1]  # (batch_size, n_dims)
        num_dims = target_items.size(-1)
        # The prior items (features only)
        prior = torch.tensor(self.prior, dtype=torch.float, device=device)
        prior_items = prior[..., :-1]  # (batch_size, num_tasks, n_dims)
        # The prior labels
        prior_labels = prior[..., -1].long()  # (batch_size, num_tasks)

        # Convert prior labels to one-hot
        prior_labels_one_hot = torch.nn.functional.one_hot(
            prior_labels, num_classes=self.num_labels
        ).float()  # (batch_size, num_tasks, num_labels)

        # compute log likelihood that each prior item is the target item
        scaling_factor = torch.sqrt(torch.tensor(1 + self.within_class_variance, device=device))
        llk = -(num_dims/(2*self.within_class_variance)) * torch.sum((scaling_factor * target_items.unsqueeze(1) - prior_items)**2, dim=-1)

        # softmax to get posterior probabilities
        posterior = torch.nn.functional.softmax(llk, dim=-1) #TODO: change if using non-uniform prior

        # Expand posterior so we can multiply by one-hot labels
        posterior = posterior.unsqueeze(-1)  # (batch_size, num_tasks, 1)

        # Weighted average over the one-hot labels
        posterior_predictive = torch.sum(posterior * prior_labels_one_hot, dim=1)  # (batch_size, num_labels)

        # clamp posterior predictive to avoid log(0)
        posterior_predictive = torch.clamp(posterior_predictive, min=1e-5, max=1-1e-5)
        log_posterior_predictive = torch.log(posterior_predictive)  # (batch_size, num_labels)

        # Detach if on gpu to avoid potential memory issues
        if device.type == "cuda":
            log_posterior_predictive = log_posterior_predictive.detach().cpu()

        return {self.output_key: log_posterior_predictive}
    

class CopyingClassification:
    def __init__(self, prior, within_class_variance, num_labels):
        self.output_key = "logits"
        self.labels_required_for_inference = False
        self.prior = prior
        self.within_class_variance = within_class_variance
        self.num_labels = num_labels

    @property
    def device(self):
        # return gpu if available, otherwise cpu
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def __call__(self, pairs: torch.Tensor) -> dict:
        """
        Returns the label for the target task based on similarity to context items.
        
        Args:
            pairs: item-label pairs tensor of shape (batch_size, context_length, n_dims+1).

        Returns:
            dict: Contains 'logits' key with tensor of shape (batch_size, 2) representing
                 log probabilities for binary classification.
        """
        device = pairs.device
        # The target item (features only)
        target_items = pairs[:, -1, :-1]  # (batch_size, n_dims)
        num_dims = target_items.size(-1)
        # The context items (features only)
        context_items = pairs[:, :-1, :-1]  # (batch_size, context_length-1, n_dims)
        # The context labels
        context_labels = pairs[:, :-1, -1].long()  # (batch_size, context_length-1)

        # Convert context labels to one-hot
        context_labels_one_hot = torch.nn.functional.one_hot(
            context_labels, num_classes=self.num_labels
        ).float()  # (batch_size, context_length-1, num_labels)

        # compute log likelihood that each context item is the target item
        llk = -(num_dims*(1+self.within_class_variance)**2)/((2*self.within_class_variance)*(2+self.within_class_variance)) * torch.sum((target_items.unsqueeze(1) - context_items/(1+self.within_class_variance))**2, dim=-1)

        # Softmax to get posterior probabilities across context items
        posterior = torch.nn.functional.softmax(llk, dim=-1)  # (batch_size, context_length-1) #TODO: change if using non-uniform prior

        # Expand posterior so we can multiply by one-hot labels
        posterior = posterior.unsqueeze(-1)  # (batch_size, context_length-1, 1)

        # Weighted average over the one-hot labels
        posterior_predictive = torch.sum(posterior * context_labels_one_hot, dim=1)  # (batch_size, num_labels)

        # clamp posterior predictive to avoid log(0)
        posterior_predictive = torch.clamp(posterior_predictive, min=1e-5, max=1-1e-5)
        log_posterior_predictive = torch.log(posterior_predictive)  # (batch_size, num_labels)

        # Detacz if on gpu to avoid potential memory issues
        if device.type == "cuda":
            log_posterior_predictive = log_posterior_predictive.detach().cpu()

        return {self.output_key: log_posterior_predictive}

class OptimalConstantClassification:
    def __init__(self, prior, within_class_variance, num_labels):
        self.output_key = "logits"
        self.labels_required_for_inference = False
        self.prior = prior
        self.within_class_variance = within_class_variance
        self.num_labels = num_labels

    @property
    def device(self):
        # return gpu if available, otherwise cpu
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, pairs):
        device = pairs.device
        # The target item (features only)
        target_items = pairs[:, -1, :-1]  # (batch_size, n_dims)
        constant_probs = torch.ones(target_items.shape[0], self.num_labels, device=device) / self.num_labels
        log_constant_probs = torch.log(constant_probs)
        return {self.output_key: log_constant_probs}

# Solutions to regression task

class dMMSERegression:
    def __init__(self, prior, noise_variance, d_out=1):
        """
        Bayesian autoregressive predictor using a discrete prior over candidate
        weights, with posterior updates done in log-space to avoid underflow.
        
        Parameters
        ----------
        prior : torch.Tensor
            A set of candidate weight matrices, shape [N, d_in, d_out].
            If d_out=1 and it's [N, d_in], we unsqueeze to [N, d_in, 1].
        noise_variance : float
            Gaussian noise variance (sigma^2) used in the likelihood.
        d_out : int, default=1
            Output dimension.
        """
        self.output_key = "predictions"
        self.labels_required_for_inference = True

        # Ensure prior has shape [N, d_in, d_out] if d_out=1
        if prior.ndim == 2 and d_out == 1:
            prior = torch.as_tensor(prior, dtype=torch.float32).unsqueeze(-1)

        self.prior = prior  # discrete set of candidate weights
        self.noise_variance = noise_variance
        self.d_out = d_out

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, xs: torch.Tensor, labels: torch.Tensor):
        """
        Bayesian autoregressive predictor using log-space posterior updates.

        At each time step t:
        1) Compute the log-likelihood of data up to t-1 for each candidate weight W_j.
        2) Add log-prior to get unnormalized log-posterior.
        3) Use logsumexp to normalize -> log-posterior.
        4) Exponentiate -> posterior, and use it to form a weighted prediction for x_t.

        Parameters
        ----------
        xs : torch.Tensor
            [B, T, d_in]
        labels : torch.Tensor
            [B, T, d_out], or [B, T] if d_out=1
        
        Returns
        -------
        dict:
            {
               "predictions": [B, T, d_out]  # or [B, T] if d_out=1 and you want to squeeze
            }
        """
        device = xs.device
        dtype = xs.dtype

        B, T, d_in = xs.shape
        d_out = self.d_out

        # If labels is [B, T] but d_out=1, unsqueeze to [B, T, 1]
        if d_out == 1 and labels.ndim == 2:
            labels = labels.unsqueeze(-1)

        # Move prior to device/dtype => shape [N, d_in, d_out]
        W = torch.as_tensor(self.prior, dtype=dtype, device=device)
        N = W.shape[0]

        # Uniform log-prior: log(1/N) = -log(N)
        # If you had non-uniform priors, you'd have a log-prior vector of shape [N].
        log_prior = -torch.log(torch.tensor(float(N), dtype=dtype, device=device))

        # We'll store predictions in [B, T, d_out]
        predictions = torch.zeros(B, T, d_out, dtype=dtype, device=device)
        
        for t in range(T):
            if t == 0:
                # No data observed -> posterior = prior => uniform
                x_t = xs[:, t, :]  # [B, d_in]
                
                # predicted_y_j => [B, N, d_out]
                predicted_y_j = torch.einsum('bd,ndf->bnf', x_t, W)
                # Weighted average (uniform => mean over j dimension)
                y_t_pred = predicted_y_j.mean(dim=1)  # [B, d_out]

            else:
                # Observed data up to t-1
                X_obs = xs[:, :t, :]      # [B, t, d_in]
                Y_obs = labels[:, :t, :] # [B, t, d_out]

                # predicted_Y_obs_j => [B, t, N, d_out]
                predicted_Y_obs_j = torch.einsum('btd,ndf->btnf', X_obs, W)
                
                # SSE_j => [B, N]
                diff = Y_obs.unsqueeze(2) - predicted_Y_obs_j  # [B, t, N, d_out]
                squared_error = diff.pow(2).sum(dim=-1)        # sum over d_out => [B, t, N]
                sse = squared_error.sum(dim=1)                 # sum over time => [B, N]

                # log-likelihood => [B, N], ignoring additive constants
                ll = - sse / (2.0 * self.noise_variance)

                # log_unnorm_post = ll + log_prior
                # shape => [B, N], since log_prior is scalar
                log_unnorm_post = ll + log_prior

                # log-sum-exp across N => shape [B, 1]
                logsumexp = torch.logsumexp(log_unnorm_post, dim=1, keepdim=True)

                # log_posterior => [B, N]
                log_posterior = log_unnorm_post - logsumexp

                # posterior => exponentiate => [B, N]
                posterior = log_posterior.exp()

                # Weighted prediction for x_t:
                x_t = xs[:, t, :]
                predicted_y_j = torch.einsum('bd,ndf->bnf', x_t, W)  # [B, N, d_out]

                # Weighted sum across N => [B, d_out]
                y_t_pred = (posterior.unsqueeze(-1) * predicted_y_j).sum(dim=1)

            predictions[:, t, :] = y_t_pred
            

        # Optionally squeeze the final dimension if d_out == 1
        return {self.output_key: predictions.squeeze(-1) if d_out == 1 else predictions}

class RidgeRegression:
    def __init__(self, prior, noise_variance, d_out=1):
        """
        Autoregressive Ridge Regression.

        Parameters
        ----------
        prior : tuple or list
            Contains information for the ridge prior:
            - prior[0] is some default mean or offset,
            - prior[1] is used to compute alpha = noise_variance / prior[1].
        noise_variance : float
            Noise variance for the ridge regression formula.
        d_out : int, default=1
            Output dimension.
        """
        self.output_key = "predictions"
        self.labels_required_for_inference = True
        self.prior = prior
        self.noise_variance = noise_variance
        self.d_out = d_out

    @property
    def device(self):
        # return gpu if available, otherwise cpu
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(
        self,
        xs: torch.Tensor, 
        labels: torch.Tensor
    ):
        """
        For each time step t in a sequence, fit a ridge regression model on 
        all previous (x_i, y_i) pairs (i < t) and predict y_t for x_t.

        This is fully vectorized for a batch of sequences.

        Parameters
        ----------
        xs : torch.Tensor
            Input features for each time step, shape [B, T, d_in].
        labels : torch.Tensor
            Targets for each time step. 
            Can be [B, T, d_out] or [B, T] if d_out=1.
        Returns
        -------
        dict
            {
               'predictions': [B, T, d_out],  # or squeezed to [B, T] if d_out=1
               'weights': [B, T, d_in, d_out] # the fitted weights for each time
            }
        """

        device = xs.device
        dtype  = xs.dtype

        # Ensure labels has shape [B, T, d_out] if d_out=1
        # (Sometimes the final dimension is squeezed if it's 1)
        if self.d_out == 1 and labels.ndim == 2:
            labels = labels.unsqueeze(-1)

        batch_size, T, d_in = xs.shape
        d_out = self.d_out

        alpha = self.noise_variance / self.prior[1]

        # For the ridge term alpha * I
        I = torch.eye(d_in, dtype=dtype, device=device).unsqueeze(0)  # [1, d_in, d_in]

        # We will collect predictions for each time step in a list
        # and also keep track of the fitted weights
        predictions = []
        for t in range(T):
            if t == 0:
                # No prior data -> predict mean of distribution
                # shape => [B, d_out]
                y_pred_t = torch.full(
                    (batch_size, d_out), 
                    fill_value=self.prior[0], 
                    dtype=dtype, 
                    device=device
                )

                # If d_out=1, weights => [B, d_in]
                # Else, weights => [B, d_in, d_out]
            else:
                # Gather data up to but not including time t
                X_t = xs[:, :t, :]     # [B, t, d_in]
                Y_t = labels[:, :t, :] # [B, t, d_out]

                # Compute (X^T X + alpha * I) for each sequence in the batch
                XtX = torch.bmm(X_t.transpose(1, 2), X_t)  # [B, d_in, d_in]
                XtX_reg = XtX + alpha * I                  # [B, d_in, d_in]

                # Invert
                invXtX = torch.linalg.inv(XtX_reg)         # [B, d_in, d_in]

                # Compute W_t = inv(X^T X + alpha I) * (X^T Y)
                XtY = torch.bmm(X_t.transpose(1, 2), Y_t)  # [B, d_in, d_out]
                W_t = torch.bmm(invXtX, XtY)               # [B, d_in, d_out]

                # Predict y_t = x_t * W_t
                x_t = xs[:, t, :].unsqueeze(1)             # [B, 1, d_in]
                y_pred_t = torch.bmm(x_t, W_t).squeeze(1)  # [B, d_out]

            predictions.append(y_pred_t)

        # Stack predictions along time axis: shape => [B, T, d_out]
        predictions = torch.stack(predictions, dim=1)  # [B, T, d_out]

        return {
            self.output_key: predictions.squeeze(-1) if (d_out == 1) else predictions,
        }

class OptimalConstantLinearRegression:
    def __init__(self, prior, noise_variance):
        self.output_key = "predictions"
        self.labels_required_for_inference = True
        self.prior = prior
        self.noise_variance = noise_variance

    @property
    def device(self):
        # return gpu if available, otherwise cpu
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, xs, labels):
        return {self.output_key: torch.full_like(labels, self.prior[0], device=self.device)}
    

########################################
# Models to describe transformer behavior 
########################################

class HierarchicalBayesianModel(nn.Module):
    def __init__(self, params_init, complexity_lookup, avg_losses_lookup, metric_name, context_dependent_weights=False,
                 batch_size=None):
        """
        Args:
            params_init (dict): Dictionary of parameters to initialize the model.
            complexity_lookup (dict): Pre-computed lookup table for complexity values.
            avg_losses_lookup (dict): Pre-computed lookup table for average losses.
            metric_name (str): Name of the metric to use for evaluation.
            context_dependent_weights (bool): Whether to use context-dependent weights.
            batch_size (int): Batch size.
        """
        super().__init__()
                
        # Initialize parameters from params_init dictionary
        self.params = nn.ParameterDict()
        for param_name, param_value in params_init.items():
            self.params[param_name] = nn.Parameter(torch.as_tensor(param_value, dtype=torch.float))
        self.metric_name = metric_name
        self.context_dependent_weights = context_dependent_weights
        
        # Store pre-computed lookup tables
        self.complexity_lookup = complexity_lookup
        self.avg_losses_lookup = avg_losses_lookup

        self.batch_size = batch_size

    def calculate_weights(self, avg_losses, complexity_values, n, context_length=None, sample_losses=None):
        """
        Batched computation of weights from posteriors using the formula.
        
        Args:
            avg_losses (torch.Tensor): Tensor of shape [num_algorithms] with average losses.
            complexity_values (torch.Tensor): Tensor of shape [num_algorithms] with complexity values.
            n (int): Number of samples.
            context_length (int): Length of the context.
            sample_losses (torch.Tensor): Tensor of shape [B, context_length, num_algorithms] (per-sample losses).
        Returns:
            weights (torch.Tensor): Tensor of shape [B, context_length, num_algorithms] with weights that sum to 1 for each sample.
        """
        # Compute posteriors
        log_priors = -torch.log(torch.tensor(2.0))*torch.pow(complexity_values.unsqueeze(0).unsqueeze(0),self.params["log_beta"].exp()) # [1, 1, num_algorithms]
        if self.context_dependent_weights:
            log_likelihoods = -self.params["log_gamma"].exp()*torch.pow(n*self.batch_size*context_length, 1-self.params["log_alpha"].exp())*avg_losses.unsqueeze(0).unsqueeze(0) # [1, 1, num_algorithms]
        else:
            log_likelihoods = -self.params["log_gamma"].exp()*torch.pow(n, 1-self.params["log_alpha"].exp())*avg_losses.unsqueeze(0).unsqueeze(0) # [1, 1, num_algorithms]
        # compute context-dependent weights if requested
        if self.context_dependent_weights and sample_losses is not None:
            cumulative_context_losses = sample_losses.cumsum(dim=1) # [B, context_length, num_algorithms]
            context_lengths = torch.arange(1, context_length+1, device=cumulative_context_losses.device, dtype=torch.float).unsqueeze(0).unsqueeze(-1) # [1, context_length, 1]
            avg_context_losses = cumulative_context_losses / context_lengths # [B, context_length, num_algorithms]
            context_dependent_term = -self.params["log_gamma"].exp()*torch.pow(context_lengths, 1-self.params["log_alpha"].exp())*avg_context_losses # [B, context_length, num_algorithms]
            log_likelihoods = log_likelihoods + context_dependent_term # [B, context_length, num_algorithms]
        
        log_posteriors = log_priors + log_likelihoods # [B, context_length, num_algorithms]

        if log_posteriors.shape[-1] == 2:  # if there are only two algorithms, we can compute the prior and likelihood odds and posterior odds
            log_prior_odds = (log_priors[..., 0] - log_priors[..., 1]).squeeze() 
            log_bayes_factor = (log_likelihoods[..., 0] - log_likelihoods[..., 1]).squeeze() 
            log_posterior_odds = log_posteriors[..., 0] - log_posteriors[..., 1]
        else:
            log_prior_odds = log_priors.squeeze()
            log_bayes_factor = log_likelihoods.squeeze()
            log_posterior_odds = log_posteriors.squeeze()

        weights = log_posteriors.softmax(dim=-1) # [B, context_length, num_algorithms]

        return weights, log_prior_odds, log_bayes_factor, log_posterior_odds
    
    def forward(self, algo_outs, algo_losses, transformer_outs, exp_params):
        """
        Forward pass for a batch.
        
        Args:
            algo_outs (torch.Tensor): Algorithm outputs of shape (context_length, num_dims, num_algorithms) or (B, context_length, num_dims, num_algorithms).
            algo_losses (torch.Tensor): Algorithm losses of shape (context_length, num_algorithms) or (B, context_length, num_algorithms).
            transformer_outs (torch.Tensor): Transformer output distributions of shape (B, context_length, num_dims).
            exp_params (dict): Dictionary with experimental parameter values for each sample.
                               Expected keys: "checkpoint", "num_tasks", "num_dims", "context_length".
        Returns:
            dict: Contains loss, weights, interpolated_out, params
        """
        device = transformer_outs.device

        n, num_tasks, num_dims, context_length = exp_params["checkpoint"], exp_params["num_tasks"], exp_params["num_dims"], exp_params["context_length"]

        # Use pre-computed lookup tables
        key = (num_tasks, num_dims, context_length)
        complexity_values = self.complexity_lookup[key].to(device)
        avg_losses = self.avg_losses_lookup[key].to(device)


        weights, log_prior_odds, log_bayes_factor, log_posterior_odds = self.calculate_weights(
            avg_losses=avg_losses, complexity_values=complexity_values, n=n, 
            context_length=context_length, sample_losses=algo_losses)
        interpolated_out = (weights.unsqueeze(-2) * algo_outs).sum(dim=-1) # [B, context_length, num_dims, 1] or [B, context_length, 1]
        
        if len(interpolated_out.shape) == 4:
            interpolated_out = interpolated_out.squeeze(-1) # [B, context_length, num_dims]
        
        if self.metric_name == "MSE":
            # Compute the final MSE between the interpolated output and transformer output
            loss = nn.MSELoss()(interpolated_out, transformer_outs).mean()
        else:
            # Compute the final KL divergence between the interpolated output and transformer output
            loss = nn.KLDivLoss(reduction='none')(torch.log(interpolated_out), transformer_outs).sum(dim=-1).mean()

        return {
            "loss": loss,
            "weights": weights,
            "interpolated_out": interpolated_out,
            "params": self.params,
            "log_prior_odds": log_prior_odds,
            "log_bayes_factor": log_bayes_factor,
            "log_posterior_odds": log_posterior_odds.mean()
            }
    

class HierarchicalBayesianModelAblated(HierarchicalBayesianModel):
    def __init__(self, params_init, complexity_lookup, avg_losses_lookup, metric_name, 
                 context_dependent_weights=False, linear_likelihood=False, fixed_complexity=False, no_random_loss_term=False,
                 linear_param_on_complexity=False, batch_size=None):
        """
        Args:
            params_init (dict): Dictionary of parameters to initialize the model.
            complexity_lookup (dict): Pre-computed lookup table for complexity values.
            avg_losses_lookup (dict): Pre-computed lookup table for average losses.
            metric_name (str): Name of the metric to use for evaluation.
            context_dependent_weights (bool): Whether to use context-dependent weights.
            linear_likelihood (bool): Whether to use linear likelihood.
            fixed_complexity (bool): Whether to use fixed complexity.
            no_random_loss_term (bool): Whether to remove random loss term.
            linear_param_on_complexity (bool): Whether to use linear parameter on complexity.
            batch_size (int): Batch size.
        """
        super().__init__(params_init, complexity_lookup, avg_losses_lookup, metric_name, context_dependent_weights, batch_size)
        if linear_likelihood:
            # set log_alpha to be a constant that is not learned 
            self.params["log_alpha"] = nn.Parameter(torch.as_tensor(-1e10, dtype=torch.float), requires_grad=False)
            print("log_alpha is now a constant that is not updated with value: ", self.params["log_alpha"])
        if no_random_loss_term:
            # set log_gamma to be a constant that is not learned
            self.params["log_gamma"] = nn.Parameter(torch.as_tensor(0.0, dtype=torch.float), requires_grad=False)
            print("log_gamma is now a constant that is not updated with value: ", self.params["log_gamma"])
        if fixed_complexity:
            # set log_beta to be a constant that is not learned
            self.params["log_beta"] = nn.Parameter(torch.as_tensor(0.0, dtype=torch.float), requires_grad=False)
            print("log_beta is now a constant that is not updated with value: ", self.params["log_beta"])
        self.linear_param_on_complexity = linear_param_on_complexity

    def calculate_weights(self, avg_losses, complexity_values, n, context_length=None, sample_losses=None):
        """
        Batched computation of weights from posteriors using the formula.
        
        Args:
            avg_losses (torch.Tensor): Tensor of shape [num_algorithms] with average losses.
            complexity_values (torch.Tensor): Tensor of shape [num_algorithms] with complexity values.
            n (int): Number of samples.
            context_length (int): Length of the context.
            sample_losses (torch.Tensor): Tensor of shape [B, context_length, num_algorithms] (per-sample losses).
        Returns:
            weights (torch.Tensor): Tensor of shape [B, context_length, num_algorithms] with weights that sum to 1 for each sample.
        """
        # Compute posteriors
        if self.linear_param_on_complexity:
            log_priors = -torch.log(torch.tensor(2.0))*complexity_values.unsqueeze(0).unsqueeze(0)*self.params["log_beta"].exp() # [1, 1, num_algorithms]
        else:
            log_priors = -torch.log(torch.tensor(2.0))*torch.pow(complexity_values.unsqueeze(0).unsqueeze(0),self.params["log_beta"].exp()) # [1, 1, num_algorithms]
        if self.context_dependent_weights:
            log_likelihoods = -self.params["log_gamma"].exp()*torch.pow(n*self.batch_size*context_length, 1-self.params["log_alpha"].exp())*avg_losses.unsqueeze(0).unsqueeze(0) # [1, 1, num_algorithms]
        else:
            log_likelihoods = -self.params["log_gamma"].exp()*torch.pow(n, 1-self.params["log_alpha"].exp())*avg_losses.unsqueeze(0).unsqueeze(0) # [1, 1, num_algorithms]
        # compute context-dependent weights if requested
        if self.context_dependent_weights and sample_losses is not None:
            cumulative_context_losses = sample_losses.cumsum(dim=1) # [B, context_length, num_algorithms]
            context_lengths = torch.arange(1, context_length+1, device=cumulative_context_losses.device, dtype=torch.float).unsqueeze(0).unsqueeze(-1) # [1, context_length, 1]
            avg_context_losses = cumulative_context_losses / context_lengths # [B, context_length, num_algorithms]
            context_dependent_term = -self.params["log_gamma"].exp()*torch.pow(context_lengths, 1-self.params["log_alpha"].exp())*avg_context_losses # [B, context_length, num_algorithms]
            log_likelihoods = log_likelihoods + context_dependent_term # [B, context_length, num_algorithms]
        
        log_posteriors = log_priors + log_likelihoods # [B, context_length, num_algorithms]

        if log_posteriors.shape[-1] == 2:  # if there are only two algorithms, we can compute the prior and likelihood odds and posterior odds
            log_prior_odds = (log_priors[..., 0] - log_priors[..., 1]).squeeze() 
            log_bayes_factor = (log_likelihoods[..., 0] - log_likelihoods[..., 1]).squeeze() 
            log_posterior_odds = log_posteriors[..., 0] - log_posteriors[..., 1]
        else:
            log_prior_odds = log_priors.squeeze()
            log_bayes_factor = log_likelihoods.squeeze()
            log_posterior_odds = log_posteriors.squeeze()

        weights = log_posteriors.softmax(dim=-1) # [B, context_length, num_algorithms]

        return weights, log_prior_odds, log_bayes_factor, log_posterior_odds