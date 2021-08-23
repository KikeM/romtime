from romtime.utils import compute_rom_difference

import numpy as np


def test_compute_rom_difference():

    uN = np.array([1.0, 2.0, 3.0, 4.0])
    uN_hat = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    basis_hat = np.random.rand(10, 5)

    norm = compute_rom_difference(uN=uN, uN_srom=uN_hat, V_srom=basis_hat)

    last = uN_hat[-1] * basis_hat[:, -1]
    expected = np.linalg.norm(last)
    N = basis_hat.shape[0]
    expected /= np.sqrt(N)

    assert np.isclose(expected, norm)


def test_compute_rom_difference_extra():

    uN = np.array([1.0, 2.0, 3.0, 4.0])
    uN_hat = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    basis_hat = np.random.rand(10, 7)

    norm = compute_rom_difference(uN=uN, uN_srom=uN_hat, V_srom=basis_hat)

    last = uN_hat[-1] * basis_hat[:, -1]
    last += uN_hat[-2] * basis_hat[:, -2]
    last += uN_hat[-3] * basis_hat[:, -3]

    expected = np.linalg.norm(last)
    N = basis_hat.shape[0]
    expected /= np.sqrt(N)

    assert np.isclose(expected, norm)