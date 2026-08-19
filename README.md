# pyturb

A microstructure processing toolbox for Rockland Scientific microstructure instruments.

## Installation

Install using `pip`. 

## Usage

```mermaid
flowchart TD;
    pfile[L0: .p] -->|pyturb p2nc| ncfile[L1: .nc];
    ncfile --- C[ ]:::empty;
    glider[glider.data.nc] --- C;
    C -->|pyturb eps| l2[L2: *_0001.nc, ... *_N.nc];
    l2 -->|pyturb bin| l3[L3: .binned.nc];
    
classDef empty fill:none,stroke:none,color:transparent,width:1px,height:1px;
```

Pyturb is primarily a CLI. The processing should be run in the following order:

1. Convert p files to netCDF using `pyturb p2nc`. Optionally, merge converted p files with `pyturb merge`.
2. Calculate turbulence estimates per-profile using `pyturb eps`.
3. Bin estimates onto a regular grid with `pyturb bin`.

### `p2nc` - convert P files

Convert Rockland binary P-files to NetCDF format:

```bash
pyturb p2nc ./path/to/raw_data/*.p -o ./converted/
```

Note that unlike the ODAS toolbox, this conversion does not apply a velocity scaling to the microstructure shear or temperature gradient. Consequently, the units of these variables are different to their ODAS counterparts. The scaling is applied later.

The merge utility enables merging of netcdf files (e.g. `pyturb merge -o ./merged/merged.nc ./converted/*.nc`). This may be useful in the case where profiles are split across multiple files and per-file processing would result in incomplete profiles.

### `eps` - calculate the dissipation rate

Estimate turbulent kinetic energy dissipation rate from converted NetCDF files:

```bash
pyturb eps ./converted/*.nc -o ./eps_output/
```

The `eps` command automatically detects multiple profiles within each input file. Output files are named `{input_stem}_p{NNNN}.nc`. Data from other instruments may be merged at this step to improve the calculations. For example, temperature and salinity may be merged from a Slocum glider and used to esimate viscosity. Velocity from a calibrated glider flight model may also be used.

A selection of the option:
- `--diss-len`: Dissipation window length in seconds (default: 4.0)
- `--fft-len`: FFT segment length in seconds (default: 1.0)  
- `--min-speed`: Minimum speed threshold in m/s (default: 0.2)
- `--direction`: Profile direction to process: `down`, `up`, or `both` (default: down)
- `--peaks-height`: Minimum peak height for profile detection in dbar (default: 25.0). Relies on [profinder](github.com/oceancascades/profinder.git)
- `--aux`: Auxiliary NetCDF file with platform data (e.g. glider lat, lon, T, S)
- `--thermo`/`--no-thermo`: Compute additional thermodynamic variables with gsw.
- `--match-conductivity`/`--no-match-conductivity`: Apply lag corrections for conductivity and temperature.
- `--skip-existing`/`--no-skip-existing`: Skip a file entirely if any output already exists for its stem. Ignored with `--overwrite`.
- `--stationary-platform`/`--moving-platform`: Use one lat/lon per profile instead of interpolating a position onto every window/bin. Default: auto-detected from the p-file's `vehicle` field (`vmp`/`rvmp`/`xmp` are treated as stationary; anything else is treated as moving, e.g. gliders). For a stationary platform, `lat`/`lon` are written as dimensionless scalars (the position at the profile's first timestamp); for a moving platform they vary with `time`/`ctd_time` as before.

CTD scalars (pressure, temperature, salinity, conductivity, density) can be attached to a finer `ctd_time` axis (`*_hires` variables, e.g. `temperature_hires`), alongside the usual dissipation-bin versions. Bin width is set by `ctd_bin_sec`. Pass `ctd_bin_sec=0` to disable. Velocity and viscosity are only meaningful at dissipation-window resolution and are not included in the `_hires` output.

Depth `z` (m, negative below the surface) is computed from pressure via `gsw.z_from_p` at both the dissipation-bin (`z`) and hires (`z_hires`) resolutions. When `--thermo` is enabled, buoyancy frequency squared `N2` is also computed via `gsw.Nsquared`, using the dissipation-bin-averaged (not hires) absolute salinity, conservative temperature, and pressure to reduce noise; `gsw.Nsquared` returns values at mid-point pressures, which are interpolated back onto the original pressure grid before being written as `N2`.

See `pyturb eps --help` formore details. 

Example processing just up casts:
```bash
pyturb eps -o ./eps/ --direction up ./converted/*.nc
```

### `bin` - bin average the data

Bin epsilon estimates by depth and concatenate into a single file:

```bash
pyturb bin -o binned_profiles.nc --bin-width 2.0 --dmax 500 ./eps_output/*.nc
```

A selection of the option:
- `--bin-width`: Depth bin width in meters (default: 2.0)
- `--dmin`/`--dmax`: Depth range for binning (default: 0-1000 m)
- `--ctd-bin-width`: Also bin the higher-resolution CTD variables (see `eps` above) onto a separate, typically finer grid of this width, on a separate `ctd_depth` coordinate (default: off)

Profiles are concatenated along a `profile` dimension and sorted chronologically.

## Methods

### Preprocessing

Before computing epsilon, profiles undergo:

1. Low-pass filtering of speed (or dP/dt-derived speed) to remove high-frequency noise.
2. Shear signals are scaled by 1/U^2 and temperature gradients by 1/U to convert to physical units.
3. Iterative removal of outliers from shear and temperature gradient signals using a form of median filter.

### Shear spectrum processing

The dissipation rate is estimated by fitting shear spectra to the Nasmyth spectrum:

1. Spectra are calculated using Welch's method with overlapping FFT windows. 
2. Spectra are converted to wavenumber spectra assuming Taylor's frozen turbulence hypothesis using the mean velocity over the window.
3. Single-pole transfer functions are used to correct spatial averaging of the shear probe and anti-aliasing filter (Rockland Technical Note 026).
4. Epsilon is estimated by fitting the observed spectrum to the theoretical Nasmyth spectrum in the inertial subrange.
5. Unresolved high-wavenumber variance is accounted for using the integrated Nasmyth spectrum.
6. Quality control metrics including mean absolute deivation are computed and QC flag attached. 
