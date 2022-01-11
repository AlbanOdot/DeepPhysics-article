import numpy as np
import torch
import helper
import os

# General global variable declaration
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
number_of_dofs = 12075

# Create the architecture and load the trained weights
network = helper.init_network(device=device, name="Propeller", number_of_dofs=number_of_dofs)
print(network)

# Load input forces
forces_numpy = np.load(f"{os.path.dirname(os.path.realpath(__file__))}/Input_forces.npy")
forces = torch.as_tensor(data=forces_numpy, dtype=torch.float, device=device)
forces.requires_grad = False

# Load corresponding displacements
ground_truth = np.load(f"{os.path.dirname(os.path.realpath(__file__))}/Displacements.npy")

for i, force in enumerate(forces):
    print(f"\nSample n°{i}:")

    prediction = network(force.view((1, -1))).cpu().detach().numpy()
    maximum_relative_l2_error_value = helper.maximum_relative_l2(prediction=prediction, ground_truth=ground_truth[i]) * 100
    mean_relative_l2_error_value = helper.mean_relative_l2(prediction=prediction, ground_truth=ground_truth[i]) * 100
    SNR_value = helper.signal_to_noise_ratio(prediction=prediction, ground_truth=ground_truth[i])
    print(f"\tMaximum relative L2 error  : {maximum_relative_l2_error_value} %")
    print(f"\tMean relative L2 error  : {mean_relative_l2_error_value} %")
    print(f"\tSNR  : {SNR_value}")
