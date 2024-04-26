import os
import sys
from pathlib import Path

import netCDF4
import numpy as np
from loguru import logger
from scipy import integrate, signal
from tqdm import tqdm

sys.path.append(os.getcwd())


from config import Config
from utils.metrics.spec_metrics import Spec_Metrics
from utils.metrics.spectra_info import correct_spec_direction, get_cdip_buyo_freq
from utils.plot.metrics_plotter import plot_hist2d
from utils.plot.polar_spectrum import plot_spec, plot_spec_list
from utils.preprocess.extract.spec_matcher import Spec_Matcher
from utils.preprocess.interpolate.MEM import maximum_entropy_method


class CDIP_Spec_Refactor:
    """
    According to the CDIP Fouriers, it is reconstructed into a two-dimensional ocean wave spectrum.
    """

    def __init__(self, config=Config()):
        self.config = config

    def get_construct_cdip_spectra_from_spec1d(self, station, directions=72):
        cdip_spec_2d_path = f"{self.config.processed_data_dir}/CDIP/filter/MEM_dir{directions}/{station}.npy"
        if os.path.exists(cdip_spec_2d_path):
            logger.success(f"load cdip {station} data from exist file")
        else:
            logger.info(f"start process {station} cdip data using MEM")
            Path(cdip_spec_2d_path).parent.mkdir(parents=True, exist_ok=True)
            self.construct_cdip_spectra(station, cdip_spec_2d_path, directions)
        return np.load(cdip_spec_2d_path)

    def construct_cdip_spectra(self, station, cdip_spec_2d_path, directions=72):
        cdip_index_list, _ = Spec_Matcher(station).get_filter_data_index()
        filename = (
            f"{self.config.raw_data_dir}/CDIP/{station}/{station[-3:]}p1_historic.nc"
        )
        nc = netCDF4.Dataset(filename)
        cdip_spec_1d = nc.variables["waveEnergyDensity"][cdip_index_list]
        cdip_wave_a1 = nc.variables["waveA1Value"][cdip_index_list]
        cdip_wave_a2 = nc.variables["waveA2Value"][cdip_index_list]
        cdip_wave_b1 = nc.variables["waveB1Value"][cdip_index_list]
        cdip_wave_b2 = nc.variables["waveB2Value"][cdip_index_list]
        nc.close()

        cdip_spec_2d_list = []
        for i in tqdm(range(len(cdip_spec_1d)), desc=f"start MEM......"):
            spec_2d = maximum_entropy_method(
                cdip_wave_a1[i],
                cdip_wave_b1[i],
                cdip_wave_a2[i],
                cdip_wave_b2[i],
                cdip_spec_1d[i],
                directions,
            )
            cdip_spec_2d_list.append(spec_2d)
        cdip_spec_2d_list = np.array(cdip_spec_2d_list)
        logger.info(f"end MEM......")
        logger.debug(f"cdip_spec_2d.shape: {cdip_spec_2d_list.shape}")

        np.save(cdip_spec_2d_path, cdip_spec_2d_list)
        logger.success(f"save cdip {station} data success")

    def get_smooth_spec_freq_dir_time(self, station, directions=72):
        """
        Read 2D CDIP spectra obtained using MEM interpolation and smooth in frequency/direction/time
        """
        cdip_spec_smooth_path = f"{self.config.processed_data_dir}/CDIP/filter/MEM_dir{directions}_smooth_freq_dir_time/{station}.npy"
        Path(cdip_spec_smooth_path).parent.mkdir(parents=True, exist_ok=True)

        if os.path.exists(cdip_spec_smooth_path):
            logger.success(f"load smoothing cdip {station} data from exist file")
        else:
            logger.info(f"start process {station} cdip data using MEM")
            spec_list = self.get_smooth_spec_freq_dir(station, directions)
            spec_list = self.smooth_spec_time_axis(spec_list)
            np.save(cdip_spec_smooth_path, spec_list)

        spec_list = np.load(cdip_spec_smooth_path)
        return spec_list

    def smooth_spec_time_axis(self, spec_list):
        print(f"spec_list.shape: {spec_list.shape}")

        window = np.ones((3, 1, 1), dtype=np.float32) / 3.0
        spec_list = np.pad(spec_list, ((1, 1), (0, 0), (0, 0)), mode="edge")
        print(f"spec_list.shape: {spec_list.shape}")

        smoothed_spec_list = signal.convolve(spec_list, window, mode="valid")
        print(f"smoothed_spec_list.shape: {smoothed_spec_list.shape}")

        return np.array(smoothed_spec_list)

    def get_smooth_spec_freq_dir(self, station, directions=72):
        cdip_spec_freq_dir_smooth_path = f"{self.config.processed_data_dir}/CDIP/filter/MEM_dir{directions}_smooth_freq_dir/{station}.npy"
        Path(cdip_spec_freq_dir_smooth_path).parent.mkdir(parents=True, exist_ok=True)
        if os.path.exists(cdip_spec_freq_dir_smooth_path):
            logger.success(f"load cdip {station} data from exist file")
        else:
            logger.info(f"start smooth MEM_{station} spec in freq & dir axis")
            spec_list = self.get_construct_cdip_spectra_from_spec1d(station, directions)
            spec_list = self.smooth_spec_freq_dir_axis(spec_list)
            np.save(cdip_spec_freq_dir_smooth_path, spec_list)

        spec_list = np.load(cdip_spec_freq_dir_smooth_path)
        return spec_list

    def smooth_spec_freq_dir_axis(self, spec_list):
        window = np.ones((3, 3), dtype=np.float32) / 9.0

        def smooth_single_spec(spec):
            spec = np.pad(spec, ((1, 1), (0, 0)), mode="edge")
            spec = np.pad(spec, ((0, 0), (1, 1)), mode="wrap")
            return signal.convolve2d(spec, window, mode="valid")

        smooth_spec_list = [
            smooth_single_spec(spec) for spec in tqdm(spec_list, desc="Smoothing")
        ]

        return np.array(smooth_spec_list)


if __name__ == "__main__":
    spec_refactor = CDIP_Spec_Refactor()

    stations = ["CDIP028"]  # , "CDIP045", "CDIP067", "CDIP093", "CDIP107"
    directions = 36
    for station in stations:
        # Use MEM interpolation to obtain the two-dimensional CDIP spectrum and save it.
        spec_refactor.get_construct_cdip_spectra_from_spec1d(station, directions)

        # Smooth the wave spectrum obtained by MEM interpolation in the frequency/direction/time dimension
        spec_refactor.get_smooth_spec_freq_dir_time(station, directions)

        # Compare the data wave integration parameters of CDIP and IOWAGA obtained by MEM interpolation and perform verification.
        # spec_refactor.test_spec_data(station, directions)
