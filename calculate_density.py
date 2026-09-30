#!/usr/bin/env python3
"""Density task: configure and orchestrate independent compute/average objects."""
import argparse
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message=".*seawater library is deprecated.*")

import numpy as np
import seawater as sw
import xarray as xr

from icon_pipeline import ICONPipeline, TimeSeriesAverager, ZarrIO

# Task-specific inclusive periods: integer years, YYYY-MM, or YYYY-MM-DD.
# Examples: (1950, 1954, ...), ("1950-03", "1952-08", ...).
PERIODS = (
    (1950, 1969,
     "dkrz.disk.model-output.icon-esm-er.hist-1950.v20240618.ocean.gr025.ml_monthly_mean"),
    (2031, 2050,
     "dkrz.disk.model-output.icon-esm-er.highres-future-ssp245.v20240618.ocean.gr025.model-level_monthly_mean"),
)
# These catalog entries contain monthly means. For daily/equally weighted
# samples, configure DensityTask with weighting="equal" and update the label.
NAMES = ("sigma0_eos80", "sigma2_eos80")


class DensityPipeline(ICONPipeline):

    def __init__(self, periods=PERIODS, base_dir=None, io=None):
        super().__init__(
            name="density",
            input_variables=["so", "to"],
            output_names=NAMES,
            periods=periods,
            base_dir=base_dir,
            io=io,
            time_product="monthly",
            title="ICON-ESM-ER monthly sigma0 and sigma2",
        )

    # -- Input preprocessing ------------------------------------------------

    def preprocess(self, ds):
        """Validate salinity units and convert temperature to Celsius."""
        salinity_units = str(ds.so.attrs.get("units", "")).lower()
        if salinity_units not in {"psu", "1", "1e-3", "0.001"}:
            raise ValueError(
                f"Unrecognized practical salinity units: {salinity_units}")
        ds["to"] = self.temperature_in_celsius(ds.to, None)
        return ds

    # -- Core computation ---------------------------------------------------

    @staticmethod
    def _density_block(salt, theta):
        """Calculate both densities in one task; share conversion and mask."""
        salt = np.asarray(salt, dtype=np.float64)
        theta = np.asarray(theta, dtype=np.float64)
        valid = np.isfinite(salt) & np.isfinite(theta)
        salt = np.where(valid, salt, np.nan)
        theta = np.where(valid, theta, np.nan)
        sigma0 = sw.dens0(salt, theta) - 1000.0
        theta2 = sw.ptmp(salt, theta, 0.0, 2000.0)
        sigma2 = sw.dens(salt, theta2, 2000.0) - 1000.0
        return sigma0, sigma2

    def compute(self, ds):
        s0, s2 = xr.apply_ufunc(
            self._density_block, ds.so, ds.to,
            input_core_dims=[[], []], output_core_dims=[[], []],
            dask="parallelized", output_dtypes=[np.float64, np.float64],
        )
        result = xr.Dataset(dict(zip(NAMES, (s0, s2)))).transpose(*self.DIMS)
        for name, reference in zip(NAMES, (0, 2000)):
            result[name].attrs = {
                "long_name": f"Potential density anomaly referenced to {reference} dbar",
                "units": "kg m-3", "equation_of_state": "EOS-80",
                "reference_pressure": reference, "reference_pressure_units": "dbar",
                "description": "Potential density minus 1000 kg m-3; all input model levels",
            }
        return result


class DensityTask:
    """Task-level composition; calculation and averaging remain independent."""

    def __init__(self, periods=PERIODS, base_dir=None, weighting="days_in_month"):
        self.io = ZarrIO()
        self.calculator = DensityPipeline(periods=periods, base_dir=base_dir, io=self.io)
        self.averager = TimeSeriesAverager(variables=NAMES, weighting=weighting, io=self.io)

    def mean_path(self, index):
        start, end, _ = self.calculator.periods[index]
        return self.calculator.base / "data" / f"density_icon_{start}-{end}_mean3d.zarr"

    def run(self, stage="all"):
        if stage not in {"all", "compute", "mean"}:
            raise ValueError(f"Unknown stage: {stage}")
        if stage in {"all", "compute"}:
            self.calculator.run()
        if stage in {"all", "mean"}:
            for index, (start, end, _) in enumerate(self.calculator.periods):
                _, timeseries = self.calculator.paths(index)
                self.calculator.log(f"period_{index}_mean", state="running")
                target = self.averager.average_zarr(
                    timeseries, self.mean_path(index), start=start, end=end)
                self.calculator.log(f"period_{index}_mean", state="complete", mean3d=str(target))
        self.calculator.log("pipeline_done", state="complete", stage=stage)

    def main(self):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--stage", choices=("all", "compute", "mean"), default="all")
        parser.add_argument("--workers", type=int, default=32)
        parser.add_argument("--threads", type=int, default=4)
        parser.add_argument("--memory", default="6GiB")
        args = parser.parse_args()
        from distributed import Client, LocalCluster
        scratch = (Path(os.environ.get("TMPDIR", "/tmp"))
                   / f"icon_density_{os.getuid()}_{os.environ.get('SLURM_JOB_ID', os.getpid())}")
        try:
            with LocalCluster(n_workers=args.workers, threads_per_worker=args.threads,
                              memory_limit=args.memory, local_directory=str(scratch),
                              dashboard_address=None) as cluster, Client(cluster):
                self.calculator.log("cluster_ready", workers=args.workers, threads=args.threads,
                                    memory=args.memory, scratch=str(scratch))
                self.run(stage=args.stage)
        except BaseException as exc:
            self.calculator.log("pipeline_failed", state="failed", error=repr(exc))
            raise


if __name__ == "__main__":
    DensityTask().main()
