import numpy as np
import torch
import helper
import os

# General global variable declaration
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
number_of_dofs = 12000

# Create the architecture and load the trained weights
network_LR_STAR = helper.init_network(device=device, name="Beam_LR_STAR", number_of_dofs=number_of_dofs)
print(network_LR_STAR)

network_MSE = helper.init_network(device=device, name="Beam_MSE", number_of_dofs=number_of_dofs)
print(network_MSE)

# Load input forces
forces_numpy = np.load(f"{os.path.dirname(os.path.realpath(__file__))}/Input_forces.npy")
forces = torch.as_tensor(data=forces_numpy, dtype=torch.float, device=device)
forces.requires_grad = False

# Load corresponding displacements
ground_truth = np.load(f"{os.path.dirname(os.path.realpath(__file__))}/Displacements.npy")

for i, force in enumerate(forces):
    print(f"\nSample n°{i}:")

    print("\tMSE :")
    prediction_MSE = network_MSE(force.view((1, -1))).cpu().detach().numpy()
    maximum_relative_l2_error_value_MSE = helper.maximum_relative_l2(prediction=prediction_MSE, ground_truth=ground_truth[i])
    mean_relative_l2_error_value_MSE = helper.mean_relative_l2(prediction=prediction_MSE, ground_truth=ground_truth[i])
    SNR_value_MSE = helper.signal_to_noise_ratio(prediction=prediction_MSE, ground_truth=ground_truth[i])
    print(f"\t\tMaximum relative L2 error  : {maximum_relative_l2_error_value_MSE} %")
    print(f"\t\tMean relative L2 error  : {mean_relative_l2_error_value_MSE} %")
    print(f"\t\tSNR  : {SNR_value_MSE}")

    print("\tLR_STAR :")
    prediction_LR_STAR = network_LR_STAR(force.view((1, -1))).cpu().detach().numpy()
    maximum_relative_l2_error_value = helper.maximum_relative_l2(prediction=prediction_LR_STAR, ground_truth=ground_truth[i])
    mean_relative_l2_error_value = helper.mean_relative_l2(prediction=prediction_LR_STAR, ground_truth=ground_truth[i])
    SNR_value = helper.signal_to_noise_ratio(prediction=prediction_LR_STAR, ground_truth=ground_truth[i])
    print(f"\t\tMaximum relative L2 error  : {maximum_relative_l2_error_value} %")
    print(f"\t\tMean relative L2 error  : {mean_relative_l2_error_value} %")
    print(f"\t\tSNR  : {SNR_value}")
