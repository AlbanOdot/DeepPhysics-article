import os

import numpy as np
import numpy.linalg as npl
from typing import Tuple

file_path = os.path.dirname(os.path.realpath(__file__))
pkg_path = file_path
import site

# Add the package to the user site-package then immediatly removes it after import
usr_site_pkg_path = site.getusersitepackages()
# Check for existence and removes it (or the ln -s will crash the script)
os.system(f'rm -rf {"".join((usr_site_pkg_path, os.path.sep, "DeepPhysics_data"))}')
print(f"Adding {pkg_path} to {usr_site_pkg_path} (Removed after the import)")
os.system(f'ln -s {pkg_path} {usr_site_pkg_path}')
from DeepPhysics_data.network_architecture import FCNN

print(f'Removing the created symlink {"".join((usr_site_pkg_path, os.path.sep, "DeepPhysics_data"))}')
os.system(f'rm -rf {"".join((usr_site_pkg_path, os.path.sep, "DeepPhysics_data"))}')

import torch


def init_network(device: torch.device, name: str, number_of_dofs: int = 1) -> FCNN:
    relative_network_path = f"/Trained_networks/{name}.pth"
    network_path = "".join((file_path, relative_network_path))
    # Generate the network architecture
    network = FCNN(number_of_dofs)
    # Replaces the random weights with the trained ones
    network.load_state_dict(state_dict=torch.load(network_path))
    # Sends the network on the target device.
    network.to(device=device)
    return network


def max_nodal_norm(u: np.ndarray, metric=npl.norm) -> Tuple[int, float]:
    """ return the index and value of the maximum of the array according to the metric"""
    maximum = -np.inf
    index = 0
    for i in range(u.shape[0]):
        tested = metric(u[i][:])
        if tested > maximum:
            maximum = tested
            index = i
    return index, maximum


def maximum_relative_l2(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """ return the value of the error according to the maximum relative L2 norm in the article"""
    error_vector = prediction.reshape(ground_truth.shape) - ground_truth

    _, max_displacement = max_nodal_norm(ground_truth)

    if -1e-10 < max_displacement < 1e-10:
        return np.inf

    scaled_error_vector = error_vector/max_displacement
    _, error_value = max_nodal_norm(scaled_error_vector)

    return error_value * 100


def mean_relative_l2(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """ return the value of the error according to the mean relative L2 norm in the article"""
    error_vector = prediction.reshape(ground_truth.shape) - ground_truth

    total_displacement = npl.norm(ground_truth)

    if -1e-5 < total_displacement < 1e-5:
        return np.inf

    return npl.norm(error_vector)/(total_displacement * error_vector.size) * 100


def signal_to_noise_ratio(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """ return the value of the signal to noise ratio"""
    noise_vector = prediction.reshape(ground_truth.shape) - ground_truth
    noise_amplitude = npl.norm(noise_vector)

    if -1e-5 < noise_amplitude < 1e-5:
        return np.inf

    return 10 * np.log10(npl.norm(prediction)/noise_amplitude)
