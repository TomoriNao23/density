#!/usr/bin/env python3
"""Reusable base class for computing derived variables from ICON ocean output.

Subclasses only need to implement:
  - preprocess(ds)    — input-specific unit checks / conversions
  - compute(ds)       — the actual physics (returns xr.Dataset with attrs)

Everything else (catalog I/O, Zarr lifecycle, year-by-year writing,
weighted-mean finalization, dask cluster setup) is handled here.

Example subclass (minimal):
    class MyPipeline(ICONPipeline):
        def __init__(self):
            super().__init__(name="mld", input_variables=["to", "so"],
                             output_names=("mld",), title="ICON mixed layer depth")

        def compute(self, ds):
            mld = ...  # your calculation
            return xr.Dataset({"mld": mld}).transpose(*self.DIMS)
"""
import argparse
import gc
import inspect
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import dask.array as da
import intake
import numpy as np
import xarray as xr
import zarr
from distributed import Client, LocalCluster


class ICONPipeline:
    """Base pipeline: open ICON data → compute → write monthly Zarr → weighted mean."""

    CATALOG = "https://raw.githubusercontent.com/eerie-project/intake_catalogues/main/eerie.yaml"
    PERIODS = (
        (1950, 1969,
         "dkrz.disk.model-output.icon-esm-er.hist-1950.v20240618.ocean.gr025.ml_monthly_mean"),
        (2031, 2050,
         "dkrz.disk.model-output.icon-esm-er.highres-future-ssp245.v20240618.ocean.gr025.model-level_monthly_mean"),
    )
    DIMS = ("time", "lev", "lat", "lon")
    CHUNKS = {"time": 12, "lev": 6, "lat": 90, "lon": 180}

    def __init__(self, name, input_variables, output_names, title=None, base_dir=None):
        """
        Parameters
        ----------
        name : str
            Short identifier used in file names and logs (e.g. "density").
        input_variables : list[str]
            Variables to select from the catalog (e.g. ["so", "to"]).
        output_names : tuple[str, ...]
            Names of the computed output variables (e.g. ("sigma0_eos80", "sigma2_eos80")).
        title : str, optional
            Human-readable title stored in Zarr global attrs.
        base_dir : str or Path, optional
            Working directory (default: directory of the calling script).
        """
        self.name = name
        self.input_variables = list(input_variables)
        self.output_names = tuple(output_names)
        self.title = title or f"ICON-ESM-ER monthly {', '.join(output_names)}"
        self.base = Path(base_dir) if base_dir else Path(__file__).resolve().parent

    # ------------------------------------------------------------------ #
    #  Logging                                                            #
    # ------------------------------------------------------------------ #

    def log(self, msg, **values):
        """Print a timestamped JSON log line and persist to state/."""
        entry = {"utc": datetime.now(timezone.utc).isoformat(), "msg": msg, **values}
        print(json.dumps(entry), flush=True)
        if msg:
            directory = self.base / "state"
            directory.mkdir(exist_ok=True)
            target = directory / (msg + ".json")
            temporary = target.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(entry, indent=2))
            temporary.replace(target)

    # ------------------------------------------------------------------ #
    #  Paths & helpers                                                    #
    # ------------------------------------------------------------------ #

    def paths(self, index):
        """Return (pending, final, mean) Zarr paths for the given period index."""
        start, end, _ = self.PERIODS[index]
        prefix = self.base / "data" / f"{self.name}_icon_{start}-{end}"
        return (Path(str(prefix) + "_monthly.inprogress.zarr"),
                Path(str(prefix) + "_monthly.zarr"),
                Path(str(prefix) + "_mean3d.zarr"))

    @staticmethod
    def zarr_format():
        key = ("zarr_format" if "zarr_format" in inspect.signature(xr.Dataset.to_zarr).parameters
               else "zarr_version")
        return {key: 2}

    # ------------------------------------------------------------------ #
    #  Input I/O                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def select_period(ds, start_year, end_year):
        """只操作时间坐标；不加载温盐全场。"""
        years = ds.time.dt.year
        selected = ds.sel(time=(years >= start_year) & (years <= end_year))
        index = selected.indexes["time"]
        if not index.is_monotonic_increasing or not index.is_unique:
            raise ValueError("时间必须严格递增且无重复。")
        actual = selected.time.dt.year.values * 12 + selected.time.dt.month.values - 1
        expected = np.arange(start_year * 12, (end_year + 1) * 12)
        if not np.array_equal(actual, expected):
            raise ValueError(
                f"{start_year}-{end_year}: 应有 {len(expected)} 个连续月份，"
                f"实际 {len(actual)} 个；存在缺月、重复月份或边界不完整。"
            )
        return selected

    @staticmethod
    def temperature_in_celsius(theta0, units_if_missing=None):
        """检查已声明的温度类型，并把表面参考位温统一为摄氏度。"""
        standard_name = theta0.attrs.get("standard_name", "")
        if standard_name in {"sea_water_temperature", "sea_water_conservative_temperature"}:
            raise ValueError(f"输入声明为 {standard_name}，不能直接当作表面参考位温。")
        units = theta0.attrs.get("units") or units_if_missing
        normalized = str(units).strip().lower().replace(" ", "").replace("_", "")
        celsius_units = {"degc", "c", "°c", "celsius", "degreecelsius", "degreescelsius",
                         "degreec", "degreesc"}
        kelvin_units = {"k", "kelvin", "degk", "degreekelvin", "degreeskelvin"}
        if normalized in celsius_units:
            result = theta0.copy(deep=False)
        elif normalized in kelvin_units:
            result = theta0 - 273.15
        else:
            raise ValueError(f"无法识别温度单位 {units!r}，请检查原始数据。")
        result.attrs = {**theta0.attrs, "units": "degC"}
        return result

    def open_inputs(self, index):
        """Open the catalog entry, select variables, validate, and preprocess."""
        start, end, key = self.PERIODS[index]
        raw = intake.open_catalog(self.CATALOG)[key].to_dask()
        try:
            ds = raw[self.input_variables]
            if "depth" in ds.dims:
                ds = ds.rename({"depth": "lev"})
            for variable in self.input_variables:
                if set(ds[variable].dims) != set(self.DIMS):
                    raise ValueError(f"Unexpected {variable} dimensions: {ds[variable].dims}")
            for coord in self.DIMS[1:]:
                if ds[coord].dims != (coord,):
                    raise ValueError(f"Expected a 1D coordinate: {coord}")
            ds = self.select_period(ds, start, end)
            ds = self.preprocess(ds)
            return raw, ds
        except BaseException:
            raw.close()
            raise

    # ------------------------------------------------------------------ #
    #  Hooks — override in subclasses                                     #
    # ------------------------------------------------------------------ #

    def preprocess(self, ds):
        """Override: input-specific preprocessing (unit checks, conversions, etc.).

        Called after variable selection and time-period filtering, before any
        computation.  Must return the (possibly modified) dataset.
        """
        return ds

    def compute(self, ds):
        """Override: compute output variables from input dataset.

        Must return an ``xr.Dataset`` with variables named in
        ``self.output_names``, each carrying appropriate ``attrs``.
        The dataset must be transposable to ``self.DIMS``.
        """
        raise NotImplementedError("Subclass must implement compute()")


    # ------------------------------------------------------------------ #
    #  Validation                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _clean_encoding(ds):
        ds = ds.copy(deep=False)
        for name in ds.variables:
            chunks = ds[name].chunks
            ds[name].encoding = {"chunks": tuple(c[0] for c in chunks) if chunks else ds[name].shape}
        return ds

    def ocean_sample(self, ds):
        """Extract a tiny spatial/temporal subset for fast validation."""
        return ds.isel(
            time=[0, ds.sizes["time"] - 1],
            lev=[0, 10, 35, 60, 71] if ds.sizes["lev"] >= 72 else slice(None),
            lat=slice(350, 356) if ds.sizes["lat"] >= 356 else slice(None),
            lon=slice(700, 706) if ds.sizes["lon"] >= 706 else slice(None),
        ).compute()

    def validate_sample(self, saved, inputs):
        """Generic validation: recompute from a small sample, compare with saved."""
        expected_input = self.ocean_sample(inputs)
        expected = self.compute(expected_input)
        actual = self.ocean_sample(saved)
        xr.testing.assert_allclose(actual[list(self.output_names)], expected)
        if not all(np.isfinite(actual[name]).any().item() for name in self.output_names):
            raise ValueError("No valid values in the ocean validation sample")


    # ------------------------------------------------------------------ #
    #  Pipeline                                                           #
    # ------------------------------------------------------------------ #

    def run(self):
        """Single-pass pipeline: for each period, prepare → calculate → finalize."""
        (self.base / "data").mkdir(exist_ok=True)

        for index, (start, end, key) in enumerate(self.PERIODS):
            pending, final, meanpath = self.paths(index)
            if any(p.exists() for p in (pending, final, meanpath)):
                raise FileExistsError(
                    f"Output already exists for {start}-{end}; refusing to overwrite")

            self.log(f"period_{index}_open", state="opening", period=f"{start}-{end}")
            raw, ds = self.open_inputs(index)
            began = time.monotonic()
            try:
                assert ds.sizes["time"] == 240 and ds.sizes["lev"] == 72

                # --- Validate with a small sample ---
                small = self.ocean_sample(ds)
                self.validate_sample(self.compute(small), small)
                self.log(f"period_{index}_validated", state="sample_ok",
                         period=f"{start}-{end}")

                # --- Create Zarr template (metadata only) ---
                shape = tuple(ds.sizes[d] for d in self.DIMS)
                chunks = tuple(self.CHUNKS[d] for d in self.DIMS)
                template = xr.Dataset(
                    {name: (self.DIMS, da.full(shape, np.nan, chunks=chunks))
                     for name in self.output_names},
                    coords={d: ds[d] for d in self.DIMS},
                )
                sample_result = self.compute(small)
                for name in self.output_names:
                    template[name].attrs = sample_result[name].attrs
                template.attrs = {
                    "title": self.title,
                    "source_catalog": self.CATALOG, "source_catalog_entry": key,
                    "period": f"{start}-{end}", "month_count": 240,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                }
                self._clean_encoding(template).to_zarr(
                    pending, mode="w-", compute=False,
                    consolidated=False, **self.zarr_format())

                # --- Compute year by year ---
                for year in range(start, end + 1):
                    self.log(f"year_{year}", state="running", period=f"{start}-{end}",
                             slurm_job=os.environ.get("SLURM_JOB_ID"))
                    annual = ds.sel(time=ds.time.dt.year == year).chunk(self.CHUNKS)
                    computed = self.compute(annual)
                    payload = computed.drop_vars(list(computed.coords))
                    for name in payload:
                        payload[name].encoding = {}
                    offset = 12 * (year - start)
                    payload.to_zarr(
                        pending, mode="r+",
                        region={"time": slice(offset, offset + 12)},
                        consolidated=False, **self.zarr_format())
                    with xr.open_zarr(pending, consolidated=False) as saved:
                        self.validate_sample(
                            saved.isel(time=slice(offset, offset + 12)), annual)
                    gc.collect()  # release file descriptors from zarr stores
                    self.log(f"year_{year}", state="complete", year=year,
                             elapsed_s=round(time.monotonic() - began, 1))

                # --- Finalize: consolidate + weighted mean ---
                self.log(f"period_{index}_finalize", state="running",
                         phase="mean", period=f"{start}-{end}")
                zarr.consolidate_metadata(str(pending))
                with xr.open_zarr(pending, consolidated=True) as monthly:
                    self.select_period(monthly, start, end)
                    weights = monthly.time.dt.days_in_month.astype("float64")
                    mean = (monthly[list(self.output_names)]
                            .weighted(weights)
                            .mean("time", skipna=True, keep_attrs=True))
                    mean = mean.transpose(*self.DIMS[1:])
                    mean.attrs = {
                        **monthly.attrs,
                        "temporal_product": "20-year mean of monthly values",
                        "averaging_method":
                            "days_in_month weights, renormalized over valid months",
                    }
                    for name in self.output_names:
                        mean[name].attrs["cell_methods"] = \
                            "time: mean (weighted by days in month)"
                    # Count valid months using the first output variable.
                    ref_var = self.output_names[0]
                    mean["valid_month_count"] = \
                        monthly[ref_var].count("time").astype("int16")
                    mean.valid_month_count.attrs = {
                        "units": "1", "expected_month_count": 240}
                    meanpending = meanpath.with_name(
                        meanpath.name.replace(".zarr", ".inprogress.zarr"))
                    self._clean_encoding(mean).to_zarr(
                        meanpending, mode="w-", consolidated=True,
                        **self.zarr_format())
                    with xr.open_zarr(meanpending, consolidated=True) as saved:
                        assert dict(saved.sizes) == \
                            {d: monthly.sizes[d] for d in self.DIMS[1:]}
                        pick = dict(lev=[0, 10, 35, 60, 71],
                                    lat=slice(350, 356), lon=slice(700, 706))
                        xr.testing.assert_allclose(
                            saved.isel(**pick).compute(),
                            mean.isel(**pick).compute())
                        stats = xr.Dataset({
                            f"{name}_{method}": getattr(saved[name], method)()
                            for name in self.output_names
                            for method in ("min", "max")
                        })
                        stats["min_valid_months"] = saved.valid_month_count.min()
                        stats["max_valid_months"] = saved.valid_month_count.max()
                        summary = {k: float(v)
                                   for k, v in stats.compute().data_vars.items()}
                        if any(not np.isfinite(v) for v in summary.values()):
                            raise ValueError(
                                f"Non-finite global mean statistics: {summary}")
                        if not (0 <= summary["min_valid_months"]
                                <= summary["max_valid_months"] <= 240):
                            raise ValueError("Invalid monthly coverage count")

                pending.rename(final)
                meanpending.rename(meanpath)
                self.log(f"period_{index}_done", state="complete",
                         period=f"{start}-{end}", monthly=str(final),
                         mean3d=str(meanpath), statistics=summary,
                         elapsed_s=round(time.monotonic() - began, 1))
            finally:
                raw.close()

        self.log("pipeline_done", state="complete",
                 periods=[f"{s}-{e}" for s, e, _ in self.PERIODS])

    # ------------------------------------------------------------------ #
    #  CLI entry point                                                    #
    # ------------------------------------------------------------------ #

    def main(self):
        """Parse CLI args, create a dask LocalCluster, and run the pipeline."""
        parser = argparse.ArgumentParser(
            description=f"ICON pipeline: {self.name}")
        parser.add_argument("--workers", type=int, default=32,
                            help="Number of dask workers (default: 32)")
        parser.add_argument("--threads", type=int, default=4,
                            help="Threads per dask worker (default: 4)")
        parser.add_argument("--memory", default="6GiB",
                            help="Memory limit per dask worker (default: 6GiB)")
        args = parser.parse_args()
        try:
            scratch = (
                Path(os.environ.get("TMPDIR", "/tmp"))
                / f"icon_{self.name}_{os.getuid()}"
                  f"_{os.environ.get('SLURM_JOB_ID', os.getpid())}"
            )
            with LocalCluster(
                n_workers=args.workers, threads_per_worker=args.threads,
                memory_limit=args.memory, local_directory=str(scratch),
                dashboard_address=None,
            ) as cluster, Client(cluster):
                self.log("cluster_ready", workers=args.workers,
                         threads=args.threads, memory=args.memory,
                         scratch=str(scratch))
                self.run()
        except BaseException as exc:
            self.log("pipeline_failed", state="failed", error=repr(exc))
            raise
