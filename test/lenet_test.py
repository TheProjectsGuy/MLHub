# %%
# Testing LeNet5
import torch
from mlhub.lenet import download_trained_model, LeNet5, \
    MNISTDataset, model_output_to_labels

# Testing example: need 32, 32 normalized image(s)
test_dataset = MNISTDataset(train=False)
test_sample = test_dataset[200] # (img: Tensor[1, 32, 32], label: int)

# Download the trained model
model: LeNet5 = download_trained_model()

with torch.no_grad():
    model_output = model(test_sample[0])
    label = model_output_to_labels(model_output)
    print(f"Label: {label}, true label: {test_sample[1]}")
