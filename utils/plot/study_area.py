from datetime import datetime
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os


import sys

sys.path.append(os.getcwd())

from config import Config


class StudyArea:
    def __init__(self) -> None:
        # 添加实心圆点和注释
        self.buyo_locations = {
            "028": (-118.6, 33.9),
            "045": (-117.5, 33.2),
            "067": (-119.9, 33.2),
            "093": (-117.4, 32.7),
            "107": (-119.8, 34.3),
        }

        self.input_points = {
            "1": (-121.5, 33.5),
            "2": (-121.0, 33.0),
            "3": (-120.5, 33.0),
            "4": (-120.5, 32.5),
            "5": (-120.0, 32.0),
            "6": (-119.5, 32.0),
            "7": (-119.0, 32.0),
        }

        self.all_available_points = {
            "1": (-117.5, 30.0),
            "2": (-122.3, 33.0),
            "3": (-119.0, 30.0),
            "4": (-122.5, 36.0),
            # "5": (-119.5, 32.0),
            "6": (-118.0, 31.5),
            # "7": (-120.5, 32.5),
            "8": (-117.5, 31.5),
            # "9": (-119.0, 32.0),
            "10": (-122.7, 33.5),
            "11": (-118.0, 30.0),
            # "12": (-120.5, 33.0),
            "13": (-117.5, 31.0),
            "14": (-118.5, 32.0),
            "15": (-123.0, 34.0),
            "16": (-121.0, 31.5),
            "17": (-117.0, 31.5),
            "18": (-117.0, 30.0),
            "19": (-118.5, 30.0),
            "20": (-122.0, 32.5),
            "21": (-116.8, 31.5),
            "22": (-119.0, 31.5),
            "23": (-120.0, 31.5),
            "24": (-122.5, 35.5),
            "25": (-122.0, 35.0),
            "26": (-121.3, 31.5),
            "27": (-117.0, 30.5),
            # "28": (-120.0, 32.0),
            "29": (-121.7, 32.0),
            "30": (-121.5, 34.0),
            # "31": (-121.0, 33.0),
            # "32": (-121.5, 33.5),
            "33": (-121.5, 34.5),
        }

        self.islands = {
            "Channel Islands": (-120, 33.45),
            "SRI": (-120.1, 33.7),
            "SCI": (-119.6, 33.8),
            "CI": (-118.6, 33.25),
        }


def plot_study_area(lon_start, lon_end, lat_start, lat_end, study_area=StudyArea()):
    # 从数据集中提取你需要的区域数据
    config = Config()

    ds = xr.open_dataset(f"{config.raw_data_dir}/ETOPO_2022_v1_30s_N90W180_bed.nc")

    subset = ds.sel(lon=slice(lon_start, lon_end), lat=slice(lat_start, lat_end))

    max_depth = 5000
    cust_cmap_val = 0.4
    cust_cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_cmap", [(0, "#ffffff"), (cust_cmap_val, "#3474a8"), (1.0, "#072d65")]
    )
    formatter = FuncFormatter(format_tick)

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent([lon_start, lon_end, lat_start, lat_end])  # 修改了起始经度为-122

    topo_contour = ax.contourf(
        subset.lon,
        subset.lat,
        -subset.z,
        cmap=cust_cmap,
        transform=ccrs.PlateCarree(),
        levels=np.linspace(0, max_depth, 200),
    )

    cbar = plt.colorbar(
        topo_contour,
        orientation="horizontal",
        pad=0.12,
        ax=ax,
        ticks=np.arange(0, max_depth + 1, 1000),
        aspect=25,
        shrink=0.65,
        anchor=(0.5, 0.6),
    )
    cbar.set_label("Depth (m)", size=8)
    cbar.ax.tick_params(labelsize=8, direction="in")

    ax.add_feature(cfeature.LAND, color="lightgrey")
    ax.add_feature(cfeature.COASTLINE)

    ax.set_xticks(
        np.arange(int(lon_start + 0.5), int(lon_end + 0.5), 1), crs=ccrs.PlateCarree()
    )
    ax.set_yticks(
        np.arange(int(lat_start + 0.5), int(lat_end) + 0.4, 0.5), crs=ccrs.PlateCarree()
    )
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xlabel("Lon(°E)", fontsize=8)
    ax.set_ylabel("Lat(°N)", fontsize=8)
    ax.tick_params(axis="both", labelsize=8, direction="in")
    ax.set_title(
        "Southern California Bight",
        fontdict={"fontsize": 10, "fontweight": "bold"},
        loc="center",
    )

    for place, (lon, lat) in study_area.buyo_locations.items():
        ax.scatter(
            lon,
            lat,
            color="#ffb61e",
            s=18,
            zorder=10,
            edgecolors="black",
            linewidths=0.5,
        )  # 添加实心圆点
        ax.annotate(
            place,
            (lon, lat),
            textcoords="offset points",
            xytext=(-8, -6),
            ha="center",
            fontsize=6,
        )

    for point, (lon, lat) in study_area.input_points.items():
        ax.scatter(
            lon,
            lat,
            color="#fcefe8",
            s=18,
            zorder=10,
            edgecolors="black",
            linewidths=0.5,
        )

    for point, (lon, lat) in study_area.all_available_points.items():
        ax.scatter(
            lon,
            lat,
            color="#00e500",
            s=15,
            zorder=10,
            edgecolors="black",
            linewidths=0.5,
        )

    for island, (lon, lat) in study_area.islands.items():
        # ax.text(lon, lat, island, transform=ccrs.PlateCarree())
        ax.annotate(
            island,
            (lon, lat),
            textcoords="offset points",
            xytext=(0, 0),
            ha="center",
            fontsize=6,
            fontstyle="italic",
        )

    time_stamp = datetime.now().strftime("%m%d-%H%M%S")
    plt.savefig(
        f"{os.getcwd()}/results/study_area_{time_stamp}.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close("all")


def format_tick(value, pos):
    return f"{value:.0f}" if value.is_integer() else f"{value:.1f}"


if __name__ == "__main__":

    plot_study_area(-122.5, -116.5, 30.5, 35.5)
