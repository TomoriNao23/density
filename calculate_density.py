#!/usr/bin/env python3
"""ICON EOS-80 density pipeline — computes sigma0 and sigma2 from so/to."""
import warnings
warnings.filterwarnings("ignore", message=".*seawater library is deprecated.*")

import numpy as np
import seawater as sw
import xarray as xr

from icon_pipeline import ICONPipeline

NAMES = ("sigma0_eos80", "sigma2_eos80")


class DensityPipeline(ICONPipeline):

    def __init__(self):
        super().__init__(
            name="density",
            input_variables=["so", "to"],
            output_names=NAMES,
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
                "description": "Potential density minus 1000 kg m-3; all 72 model levels",
            }
        return result


if __name__ == "__main__":
    DensityPipeline().main()
