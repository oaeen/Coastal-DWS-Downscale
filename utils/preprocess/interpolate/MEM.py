import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from netCDF4 import Dataset

sys.path.append(os.getcwd())
from config import Config
from utils.plot.polar_spectrum import plot_spec


def maximum_entropy_method(a1, b1, a2, b2, spec_1d, directions=72):
    """
    Lygre. A. and Krogstad. H. E.,
    "Maximum entropy estimation of the directional distribution in ocean wave spectra,"
    Journal of Physical Oceanography 16(12), 2052 -2060 (1986).
    """
    c1 = a1 + 1j * b1
    c2 = a2 + 1j * b2

    phi_1 = (c1 - c2 * np.conjugate(c1)) / (1 - np.abs(c1) ** 2)
    phi_2 = c2 - c1 * phi_1
    phi_e = 1 - phi_1 * np.conjugate(c1) - phi_2 * np.conjugate(c2)

    degree_per_direction = 360 / directions
    D = []
    theta_range = np.arange(0, 360, degree_per_direction) * np.pi / 180

    for theta in theta_range:
        D_theta = (
            (0.5 * np.pi**-1)
            * phi_e
            / (
                np.abs(1 - phi_1 * np.exp(-1j * theta) - phi_2 * np.exp(-2j * theta))
                ** 2
            )
        )
        D.append(D_theta)

    D = np.array(D).T
    D = np.abs(D)

    W = np.tile(spec_1d.reshape(-1, 1), (1, directions))

    spec_2d = W * D
    return spec_2d


if __name__ == "__main__":
    config = Config()

    station_id = "028"
    nc_filepath = (
        f"{config.raw_data_dir}/CDIP/CDIP{station_id}/{station_id}p1_historic.nc"
    )
    nc = Dataset(nc_filepath, "r")

    waveA1Value = nc.variables["waveA1Value"][:]
    waveA2Value = nc.variables["waveA2Value"][:]
    waveB1Value = nc.variables["waveB1Value"][:]
    waveB2Value = nc.variables["waveB2Value"][:]
    waveEnergyDensity = nc.variables["waveEnergyDensity"][:]
    waveFrequency = nc.variables["waveFrequency"][:]

    nc.close()

    idx = 1
    a1 = waveA1Value[idx]
    b1 = waveB1Value[idx]
    a2 = waveA2Value[idx]
    b2 = waveB2Value[idx]
    spec_1d = waveEnergyDensity[idx]

    spec_2d = maximum_entropy_method(a1, b1, a2, b2, spec_1d, directions=36)
    print(spec_2d.shape)

    plot_spec(spec_2d, "MEM_ver1", save_dir=f"{os.getcwd()}", cdip_buyo=True)
