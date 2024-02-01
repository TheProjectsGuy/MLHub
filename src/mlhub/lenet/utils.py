# Common utilities for LeNet (training and inference mainly)

# %%
import torch
from torch import nn
from torch import device
from typing import Optional
from torch.utils.data import DataLoader


# %%
def error_rate(model_labels: torch.Tensor, 
                target_labels: torch.Tensor) -> torch.Tensor:
    """
        Compute error rate. These are the fraction of the 
        ``model_labels`` that do not match the ``target_labels``.
        
        :param model_labels:    Labels predicted by model
        :param target_labels:   Ground-truth labels
    """
    assert model_labels.shape == target_labels.shape
    num = model_labels.shape[0]
    return torch.sum(model_labels != target_labels)/num


# %%
def model_output_to_labels(model_output: torch.Tensor) \
        -> torch.Tensor:
    """
        Convert model output (from the RBF unit) to labels using
        ``argmin``.
        
        :param model_output:    The output of the RBFLayer
        :returns:   The label (as index of least value)
    """
    if len(model_output.shape) == 1:    # Scalar
        return torch.argmin(model_output)
    else:   # Batched
        return torch.argmin(model_output, dim=1)


# %%
def model_output_to_multi_labels(model_output: torch.Tensor, 
            top_n: int = 1) -> torch.Tensor:
    """
        Same as :py:func:`model_output_to_labels`, but instead of
        ``argmin`` and returning only one label, it returns ``top_n``
        smallest values of the RBF output. This can be used for
        debugging purposes (to see what are the next most likely
        predictions, for example).
        
        :param model_output:   The output of the RBFLayer
        :param top_n:   The ``top_n`` value
        
        :returns:   The indices of the lowest ``top_n`` values
    """
    if len(model_output.shape) == 1:    # Scalar
        return torch.argsort(model_output)[:top_n]
    else:   # Batched
        return torch.argsort(model_output, dim=1)[:, :top_n]

# %%
@torch.no_grad()
def test(test_dataloader: DataLoader, model: nn.Module, 
            device: Optional[device] = None):
    """
        Test the model through a DataLoader on the test set. This
        function is mainly used for internal evaluation/validation.
        
        :param test_dataloader:
            The DataLoader to the MNISTDataset test split
        :param model:   The model to test
        :param device:  
            The device to use for testing. If ``None``, then it is
            inferred from the device of the ``model``.
    """
    num_samples = len(test_dataloader.dataset)
    model.eval()
    if device is None:  # Infer device from model's parameters
        device = torch.device(list(model.parameters())[0].device)
    # Placeholder
    model_preds = torch.empty([num_samples], device=device)
    test_preds = torch.empty([num_samples], device=device)
    bs = test_dataloader.batch_size
    for batch, test_sample in enumerate(test_dataloader):
        test_input, test_target = test_sample
        test_input = test_input.to(device)
        test_target = test_target.to(device)
        model_output = model(test_input)
        model_preds[batch*bs:(batch+1)*bs] \
                = model_output_to_labels(model_output)
        test_preds[batch*bs:(batch+1)*bs] = test_target
    er = error_rate(model_preds, test_preds)
    return er, model_preds, test_preds

