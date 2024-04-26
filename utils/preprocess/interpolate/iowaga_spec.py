import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from scipy.interpolate import RegularGridInterpolator as RGI
from tqdm import tqdm

sys.path.append(os.getcwd())
from config import Config
from utils.metrics.spec_metrics import Spec_Metrics
from utils.plot.metrics_plotter import plot_hist2d
from utils.preprocess.extract.iowaga_spec import get_filter_input_iowaga_spec
from utils.metrics.spectra_info import (
    generate_extend_dirs,
    generate_freq,
)


class IOWAGA_Spec_Interpolator:
    def __init__(self):
        pass

    def get_interpolate_target_spec(self, station, save_dir):
        spec_save_path = f"{save_dir}/{station}.npy"
        if os.path.exists(spec_save_path):
            logger.success(f"load input iowaga data from exist file {spec_save_path}")
        else:
            logger.info(f"start process {station} iowaga input data")
            Path(spec_save_path).parent.mkdir(parents=True, exist_ok=True)
            self.interpolate_spec(station, spec_save_path)

        spec_interpolate_list = np.load(spec_save_path)
        logger.info(f"{station}_interpolate.shape: {spec_interpolate_list.shape}")
        return spec_interpolate_list

    def interpolate_spec(
        self,
        station,
        interpolate_spec_save_path,
        frequency_num=64,
        direction_num=72,
    ):
        """
        Read IOWAGA data and filter the data according to iowaga_index_list
        Use cubic interpolation to process the IOWAGA data so that it has the same frequency and direction as the CDIP data
        """

        iowaga_spec_list = get_filter_input_iowaga_spec(station, station)
        iowaga_spec_list = np.concatenate(
            (iowaga_spec_list, iowaga_spec_list[:, :, 0:1]), axis=2
        )
        logger.debug(f"iowaga_spec.shape: {iowaga_spec_list.shape}")

        freq_ori = generate_freq()
        freq_new = generate_freq(new_n=frequency_num)

        dirs_ori = generate_extend_dirs(n=25)
        dirs_new = generate_extend_dirs(n=direction_num + 1)

        meshf, meshd = np.meshgrid(freq_new, dirs_new)
        interpolate_spec_list = []
        for spec in tqdm(
            iowaga_spec_list, desc=f"interpolate {station} data using cubic method ..."
        ):
            interpolator = RGI(
                (dirs_ori, freq_ori), spec.T, method="cubic", bounds_error=False
            )
            spec_interpolate = interpolator((meshd, meshf)).T
            interpolate_spec_list.append(spec_interpolate[:, :-1])

        interpolate_spec_list = np.array(interpolate_spec_list)
        logger.debug(f"iowaga_spec_interpolate.shape: {interpolate_spec_list.shape}")
        np.save(f"{interpolate_spec_save_path}", interpolate_spec_list)
        logger.success(f"save iowaga {station} data success")

    def get_bilinear_input_spec(
        self, iowaga_position, target_cdip_station, save_dir, direction_num=72
    ):
        """
        Process the IOWAGA input data using bilinear interpolation to make it the same frequency and direction as the CDIP data
        """
        spec_save_path = f"{save_dir}/{iowaga_position}_to_{target_cdip_station}.npy"
        if os.path.exists(spec_save_path):
            logger.success(f"load input iowaga data from exist file {spec_save_path}")
        else:
            logger.info(
                f"start process {iowaga_position}_to_{target_cdip_station} iowaga input data"
            )
            Path(spec_save_path).parent.mkdir(parents=True, exist_ok=True)
            self.interpolate_iowaga_spec_bilinear(
                iowaga_position,
                target_cdip_station,
                spec_save_path,
                direction_num=direction_num,
            )
        iowaga_spec_interpolate_list = np.load(spec_save_path)
        logger.info(
            f"{iowaga_position}_to_{target_cdip_station}_interpolate.shape: {iowaga_spec_interpolate_list.shape}"
        )
        return iowaga_spec_interpolate_list

    def interpolate_iowaga_spec_bilinear(
        self,
        iowaga_position,
        target_cdip_station,
        save_path,
        frequency_num=64,
        direction_num=72,
    ):
        """
        Read IOWAGA data and filter the data according to iowaga_index_list
        The IOWAGA data are processed using bilinear interpolation to have the same frequency and direction as the CDIP data
        """

        iowaga_spec_list = get_filter_input_iowaga_spec(
            iowaga_position, target_cdip_station
        )
        logger.debug(f"iowaga_spec_list.shape: {iowaga_spec_list.shape}")

        logger.info(f"interpolate using bilinear{iowaga_position}......")
        iowaga_spec_list = torch.from_numpy(iowaga_spec_list).float()
        iowaga_spec_list = iowaga_spec_list.unsqueeze(1)
        target_size = (frequency_num, direction_num)
        iowaga_spec_interpolate_list = F.interpolate(
            iowaga_spec_list, size=target_size, mode="bilinear", align_corners=False
        )
        iowaga_spec_interpolate_list = iowaga_spec_interpolate_list.squeeze(1).numpy()

        np.save(f"{save_path}", iowaga_spec_interpolate_list)
        logger.success(f"save iowaga {iowaga_position} data success")

    def test_interpolate_target_spec(self, station):
        metric_ori = Spec_Metrics(freq=36, direction=24, cdip_buyo=False)
        metric_interpolate = Spec_Metrics(freq=64, direction=72, cdip_buyo=False)

        iowaga_ori = get_filter_input_iowaga_spec(station, station)

        iowaga_interpolate = self.get_interpolate_target_spec(station)

        swh_ori, mwp_ori, mwd_ori = metric_ori.integral_predict_spec_parameters(
            iowaga_ori
        )

        (
            swh_interpolate,
            mwp_interpolate,
            mwd_interpolate,
        ) = metric_interpolate.integral_predict_spec_parameters(iowaga_interpolate)

        config = Config()
        config.y_location = f"{station}_interpolate"
        config.comment = "iowaga_ori_vs_interpolate"
        metric_ori.swh_desc["data_type"] = "swh_ori_vs_interpolate"
        metric_ori.mwp2_desc["data_type"] = "mwp2_ori_vs_interpolate"
        metric_ori.mwd_desc["data_type"] = "mwd_ori_vs_interpolate"

        # 绘制直方图
        plot_hist2d(swh_ori, swh_interpolate, metric_ori.swh_desc, config=config)
        plot_hist2d(mwp_ori, mwp_interpolate, metric_ori.mwp2_desc, config=config)
        plot_hist2d(mwd_ori, mwd_interpolate, metric_ori.mwd_desc, config=config)


if __name__ == "__main__":
    config = Config()
    spec_interpolator = IOWAGA_Spec_Interpolator()

    save_dir = f"{config.processed_data_dir}/IOWAGA/filter/interpolated_input_dir36"
    input_stations = [
        "W1215N335",
        "W1210N330",
        "W1205N330",
        "W1205N325",
        "W1200N320",
        "W1195N320",
        "W1190N320",
    ]
    target_stations = ["CDIP028", "CDIP045", "CDIP067", "CDIP093", "CDIP107"]
    for input_station in input_stations:
        for target_station in target_stations:
            spec_interpolator.get_bilinear_input_spec(
                input_station, target_station, save_dir, direction_num=36
            )
