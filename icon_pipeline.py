"""Reusable classes for calculation, time averaging, and Zarr storage.

No task configuration, physical formulas, CLI entry point, or task orchestration
belongs here. Import the classes in a task script and compose them there.
"""
import gc
import inspect
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr


class TimeSelection:
    """Shared time-coordinate validation and inclusive selection."""

    @staticmethod
    def select_period(ds, start=None, end=None):
        """Select inclusive year/month/date bounds on sorted, unique input times.

        Missing bounds use all available data. Selection uses available timestamps;
        it does not infer missing samples or require full coverage of the bounds.
        """
        if "time" not in ds.indexes:
            raise ValueError("Input must have an indexed time coordinate")
        index = ds.indexes["time"]
        if not index.is_monotonic_increasing or not index.is_unique:
            raise ValueError("时间必须严格递增且无重复。")
        selected = ds.sel(time=slice(None if start is None else str(start),
                                    None if end is None else str(end)))
        if selected.sizes["time"] == 0:
            raise ValueError(f"{start} 至 {end}: 所选时间段没有数据。")
        return selected


class ZarrIO:
    """Zarr v2 storage, encoding, and verified output lifecycle."""

    @staticmethod
    def format_options():
        key = ("zarr_format" if "zarr_format" in inspect.signature(xr.Dataset.to_zarr).parameters
               else "zarr_version")
        return {key: 2}

    @staticmethod
    def clean_encoding(ds):
        ds = ds.copy(deep=False)
        for name in ds.variables:
            chunks = ds[name].chunks
            shape = tuple(c[0] for c in chunks) if chunks else ds[name].shape
            # Integer 1 permits scalar results with older xarray / newer Zarr.
            ds[name].encoding = {"chunks": shape if shape else 1}
        return ds

    @staticmethod
    def pending_path(target):
        target = Path(target)
        return target.with_name(target.stem + ".inprogress" + target.suffix)

    @staticmethod
    def ensure_available(*paths):
        for path in paths:
            if Path(path).exists():
                raise FileExistsError(f"Output already exists; refusing to overwrite: {path}")

    @staticmethod
    def open(path, consolidated=None):
        return xr.open_zarr(path, consolidated=consolidated)

    def write(self, ds, path, *, clean=True, **options):
        data = self.clean_encoding(ds) if clean else ds
        return data.to_zarr(path, **options, **self.format_options())

    def commit(self, pending, target):
        import zarr
        self.ensure_available(target)
        zarr.consolidate_metadata(str(pending))
        Path(pending).rename(target)
        return Path(target)

    def write_verified(self, ds, target):
        """Write a complete Dataset, validate a sample, then publish its path."""
        target = Path(target)
        pending = self.pending_path(target)
        self.ensure_available(target, pending)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.write(ds, pending, mode="w-", consolidated=True)
        with self.open(pending, consolidated=True) as saved:
            if dict(saved.sizes) != dict(ds.sizes):
                raise ValueError("Saved dimensions differ from computed output")
            sample = {dim: slice(0, min(size, 3)) for dim, size in ds.sizes.items()}
            xr.testing.assert_allclose(saved.isel(**sample).compute(),
                                       ds.isel(**sample).compute())
        return self.commit(pending, target)


class TimeSeriesAverager:
    """Independent time averaging: validate inputs, select strategy, reduce.

    Call equal_mean(), monthly_mean(), or weighted_mean() explicitly, or use
    period_mean() to dispatch according to the configured weighting. Spatial
    dimensions are unrestricted and input timestamps are never resampled.
    """

    def __init__(self, variables=None, weighting="equal", io=None):
        self.variables = self._normalize_variables(variables)
        self._get_method(weighting)  # reject invalid configuration immediately
        self.weighting = weighting
        self.io = io if io is not None else ZarrIO()

    # -- Validation: all checks happen before the physical fields are reduced. --

    @staticmethod
    def _normalize_variables(variables):
        if variables is None:
            return None
        if isinstance(variables, str):
            return (variables,)
        return tuple(variables)

    def _validate_input(self, series):
        """Return the validated Dataset and resolved scientific variables."""
        if not isinstance(series, xr.Dataset):
            raise TypeError("Input must be an xarray Dataset")
        series = TimeSelection.select_period(series)
        variables = self.variables
        if variables is None:
            variables = tuple(name for name, var in series.data_vars.items()
                              if "time" in var.dims and np.issubdtype(var.dtype, np.number))
        if not variables or len(set(variables)) != len(variables):
            raise ValueError("Select at least one unique time-dependent variable")
        if "valid_time_count" in variables:
            raise ValueError("valid_time_count is reserved for output coverage")
        for name in variables:
            if name not in series.data_vars or "time" not in series[name].dims:
                raise ValueError(f"Variable {name!r} must be a time-dependent data variable")
            if not np.issubdtype(series[name].dtype, np.number):
                raise ValueError(f"Variable {name!r} must be numeric")
        return series, list(variables)

    @staticmethod
    def _validate_monthly_time(series):
        """Require calendar dates and at most one sample per calendar month."""
        try:
            month_ids = series.time.dt.year.values * 12 + series.time.dt.month.values
        except (AttributeError, TypeError) as exc:
            raise ValueError("Monthly weighting requires a calendar time coordinate") from exc
        if len(np.unique(month_ids)) != len(month_ids):
            raise ValueError("days_in_month 权重只适用于每月最多一个样本的数据。")

    @staticmethod
    def _validate_weights(series, weights):
        """Validate only the small weight array, keeping scientific data lazy."""
        if not isinstance(weights, xr.DataArray):
            raise TypeError("Weights must be an xarray DataArray")
        if weights.dims != ("time",) or "time" not in weights.indexes:
            raise ValueError("Weights must have exactly the indexed time dimension")
        if (not np.issubdtype(weights.dtype, np.number)
                or np.issubdtype(weights.dtype, np.complexfloating)):
            raise ValueError("Weights must be real numeric values")
        _, weights = xr.align(series, weights, join="exact")
        values = weights.compute().values
        if (not np.isfinite(values).all() or (values < 0).any()
                or not (values > 0).any()):
            raise ValueError("Weights must be finite, nonnegative, and have positive total")
        return weights

    def _validate_paths(self, input_path, output_path):
        source = Path(input_path).resolve()
        target = Path(output_path).resolve()
        pending = self.io.pending_path(target)
        if source == target or source in target.parents or source in pending.parents:
            raise ValueError("Output must be separate from the input Zarr store")
        self.io.ensure_available(target, pending)
        return source, target

    # -- Strategies: each public method validates before invoking the reducer. --

    def _get_method(self, weighting):
        methods = {"equal": self.equal_mean, "days_in_month": self.monthly_mean}
        if weighting not in methods:
            raise ValueError(f"Unsupported mean weighting: {weighting}")
        return methods[weighting]

    def equal_mean(self, series):
        """Give each input time equal weight; skip NaNs per variable/grid cell."""
        series, variables = self._validate_input(series)
        weights = xr.ones_like(series.time, dtype="float64")
        return self._calculate_mean(series, variables, weights, "equal weights")

    def monthly_mean(self, series):
        """Average existing monthly values using days in month; no resampling."""
        series, variables = self._validate_input(series)
        self._validate_monthly_time(series)
        weights = series.time.dt.days_in_month.astype("float64")
        return self._calculate_mean(series, variables, weights, "days_in_month weights")

    def weighted_mean(self, series, weights):
        """Use caller-supplied weights matching the selected time coordinate.

        Supply duration weights for irregular samples. Finite nonnegative
        weights are required, with at least one positive value.
        """
        series, variables = self._validate_input(series)
        weights = self._validate_weights(series, weights)
        return self._calculate_mean(series, variables, weights, "explicit weights")

    def period_mean(self, series, weights=None):
        """Dispatch to the configured strategy; explicit weights override it."""
        if weights is not None:
            return self.weighted_mean(series, weights)
        return self._get_method(self.weighting)(series)

    # -- Shared reduction/output metadata: no strategy-specific branches. --

    def _calculate_mean(self, series, variables, weights, method):
        mean = (series[variables].weighted(weights)
                .mean("time", skipna=True, keep_attrs=True))
        return self._annotate_result(mean, series, variables, method)

    @staticmethod
    def _annotate_result(mean, series, variables, method):
        method = f"{method}, renormalized over valid samples"
        mean.attrs = {
            **series.attrs,
            "temporal_product": "mean over selected input times",
            "averaging_method": method,
            "time_count": series.sizes["time"],
            "actual_time_start": str(series.time.values[0]),
            "actual_time_end": str(series.time.values[-1]),
            "period": f"{series.time.values[0]} to {series.time.values[-1]}",
        }
        for name in variables:
            mean[name].attrs = {**mean[name].attrs,
                                "cell_methods": f"time: mean ({method})"}
        # Coverage refers to the first selected variable, including zero-weight
        # times; it is not the denominator used by the weighted mean.
        ref_var = variables[0]
        mean["valid_time_count"] = series[ref_var].count("time").astype("int32")
        mean.valid_time_count.attrs = {
            "units": "1", "expected_time_count": series.sizes["time"],
            "reference_variable": ref_var,
            "long_name": "Number of non-missing input times for the reference variable",
        }
        return mean

    def average_zarr(self, input_path, output_path, *, start=None, end=None, weights=None):
        """Validate paths, select saved values, dispatch averaging, then save."""
        source, target = self._validate_paths(input_path, output_path)
        with self.io.open(source) as raw:
            selected = TimeSelection.select_period(raw, start, end)
            mean = self.period_mean(selected, weights=weights)
            mean.attrs["source_timeseries"] = str(source)
            mean.attrs["requested_start"] = "all available" if start is None else str(start)
            mean.attrs["requested_end"] = "all available" if end is None else str(end)
            return self.io.write_verified(mean, target)


class ICONPipeline:
    """Calculation template: read → preprocess → compute each time → save.

    Subclasses implement physical formulas. This class never calls an averager.
    """

    CATALOG = "https://raw.githubusercontent.com/eerie-project/intake_catalogues/main/eerie.yaml"
    DIMS = ("time", "lev", "lat", "lon")
    CHUNKS = {"time": 12, "lev": 6, "lat": 90, "lon": 180}

    def __init__(self, name, input_variables, output_names, periods,
                 title=None, base_dir=None, time_product="timeseries", io=None):
        """
        Parameters
        ----------
        name : str
            Short identifier used in file names and logs (e.g. "density").
        input_variables : list[str]
            Variables to select from the catalog (e.g. ["so", "to"]).
        output_names : tuple[str, ...]
            Names of the computed output variables (e.g. ("sigma0_eos80", "sigma2_eos80")).
        periods : sequence of (start, end, catalog_entry)
            Inclusive bounds: integer years or strings such as "1950-03" or
            "1950-03-15". Source time coordinates are preserved exactly.
        time_product : str
            Filename suffix, e.g. "monthly" or "daily"; does not resample data.
        title : str, optional
            Human-readable title stored in Zarr global attrs.
        base_dir : str or Path, optional
            Working directory (default: directory of the calling script).
        """
        self.name = name
        self.input_variables = list(input_variables)
        self.output_names = tuple(output_names)
        self.title = title or f"ICON-ESM-ER {', '.join(output_names)}"
        self.periods = tuple(periods)
        if not self.periods:
            raise ValueError("At least one period must be configured by the task")
        if not time_product or not all(c.isalnum() or c in "_-" for c in time_product):
            raise ValueError("time_product must be a filename-safe label")
        self.io = io if io is not None else ZarrIO()
        self.time_product = time_product
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
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / (msg + ".json")
            temporary = target.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(entry, indent=2))
            temporary.replace(target)

    # ------------------------------------------------------------------ #
    #  Paths & helpers                                                    #
    # ------------------------------------------------------------------ #

    def paths(self, index):
        """Return (pending, final) time-series paths for a configured period."""
        start, end, _ = self.periods[index]
        prefix = self.base / "data" / f"{self.name}_icon_{start}-{end}"
        return (Path(str(prefix) + f"_{self.time_product}.inprogress.zarr"),
                Path(str(prefix) + f"_{self.time_product}.zarr"))

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
        import intake
        start, end, key = self.periods[index]
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
            ds = TimeSelection.select_period(ds, start, end)
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

    def spatial_sample(self, ds):
        """Use bounded spatial samples for any number of model levels."""
        return ds.isel(
            lev=np.unique(np.linspace(0, ds.sizes["lev"] - 1,
                                      min(5, ds.sizes["lev"]), dtype=int)),
            lat=slice(350, 356) if ds.sizes["lat"] >= 356 else slice(0, 6),
            lon=slice(700, 706) if ds.sizes["lon"] >= 706 else slice(0, 6),
        )

    def ocean_sample(self, ds):
        """Extract a tiny spatial/temporal subset for fast validation."""
        return self.spatial_sample(ds).isel(
            time=np.unique([0, ds.sizes["time"] - 1]),
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
        """Compute/save native-time fields only; never call an averaging method."""
        import dask.array as da
        outputs = []
        (self.base / "data").mkdir(parents=True, exist_ok=True)

        for index, (start, end, key) in enumerate(self.periods):
            pending, final = self.paths(index)
            self.io.ensure_available(pending, final)

            self.log(f"period_{index}_open", state="opening", period=f"{start}-{end}")
            raw, ds = self.open_inputs(index)
            began = time.monotonic()
            try:
                time_count = ds.sizes["time"]

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
                    "period": f"{start}-{end}", "time_count": time_count,
                    "actual_time_start": str(ds.time.values[0]),
                    "actual_time_end": str(ds.time.values[-1]),
                    "temporal_product": self.time_product,
                    "time_processing": "native input timestamps; no resampling",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                }
                self.io.write(template, pending, mode="w-", compute=False,
                              consolidated=False)

                # Write time-chunk-aligned batches; partial years and final
                # short batches need no special offsets or calendar assumptions.
                batch_size = self.CHUNKS["time"]
                for offset in range(0, time_count, batch_size):
                    stop = min(offset + batch_size, time_count)
                    event = f"period_{index}_batch_{offset}"
                    self.log(event, state="running", period=f"{start}-{end}",
                             time_start=offset, time_stop=stop,
                             slurm_job=os.environ.get("SLURM_JOB_ID"))
                    batch = ds.isel(time=slice(offset, stop)).chunk(self.CHUNKS)
                    computed = self.compute(batch)
                    payload = computed.drop_vars(list(computed.coords))
                    for name in payload:
                        payload[name].encoding = {}
                    self.io.write(payload, pending, clean=False, mode="r+",
                                  region={"time": slice(offset, stop)}, consolidated=False)
                    with self.io.open(pending, consolidated=False) as saved:
                        self.validate_sample(saved.isel(time=slice(offset, stop)), batch)
                    gc.collect()
                    self.log(event, state="complete", time_start=offset, time_stop=stop,
                             elapsed_s=round(time.monotonic() - began, 1))

                # Commit the time series independently of any mean product.
                self.io.commit(pending, final)
                outputs.append(final)
                self.log(f"period_{index}_computed", state="complete",
                         period=f"{start}-{end}", timeseries=str(final),
                         elapsed_s=round(time.monotonic() - began, 1))
            finally:
                raw.close()

        self.log("compute_done", state="complete",
                 periods=[f"{s}-{e}" for s, e, _ in self.periods])
        return outputs
