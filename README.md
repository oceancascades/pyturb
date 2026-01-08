# pyturb

A microstructure processing toolbox for Rockland Scientific microstructure instruments.

## Installation

Install using `pip`. 

## Command Line Usage

The main pyturb commands are

### `p2nc` - convert P files

Convert Rockland binary P-files to NetCDF format:

```bash
pyturb p2nc ./path/to/raw_data/*.p -o ./converted/
```

Note that unlike the ODAS toolbox, this conversion does not apply a velocity scaling to the microstructure shear or temperature gradient. The units of these variables are different to their ODAS counterparts. The scaling is applied layer.

### `eps` - calculate the dissipation rate

Estimate turbulent kinetic energy dissipation rate (epsilon) from converted NetCDF files:

```bash
pyturb eps ./converted/*.nc -o ./eps_output/
```

The `eps` command automatically detects multiple profiles within each input file. Output files are named `{input_stem}_p{NNN}.nc` for each profile found.

A selection of the option used:
- `--diss-len`: Dissipation window length in seconds (default: 4.0)
- `--fft-len`: FFT segment length in seconds (default: 1.0)  
- `--min-speed`: Minimum speed threshold in m/s (default: 0.2)
- `--direction`: Profile direction to process: `down`, `up`, or `both` (default: down)
- `--peaks-height`: Minimum peak height for profile detection in dbar (default: 25.0). Relies on [profinder](github.com/oceancascades/profinder.git)
- `--aux`: Auxiliary NetCDF file with glider flight data (lat, lon, T, S)

Example processing just up casts:
```bash
pyturb eps ./converted/*.nc -o ./eps_output/ --direction up
```

### `bin` - bin average the data

Bin epsilon estimates by depth and concatenate into a single file:

```bash
pyturb bin ./eps_output/*.nc -o binned_profiles.nc --bin-width 2.0 --dmax 500
```

Options:
- `--bin-width`: Depth bin width in meters (default: 2.0)
- `--dmin`/`--dmax`: Depth range for binning (default: 0-1000 m)
- `--pressure`: Bin by pressure instead of depth

## Processing Methods

### Preprocessing Pipeline

Before computing epsilon, profiles undergo:

1. Low-pass filtering of speed (or dP/dt-derived speed) to remove high-frequency noise
2. Shear signals are scaled by 1/U² and temperature gradients by 1/U to convert to physical units
3. Iterative removal of outliers from shear and temperature gradient signals

### Shear Spectrum Processing

The dissipation rate is estimated by fitting shear spectra to the Nasmyth spectrum:

1. Shear probe signals are converted to wavenumber spectra using Welch's method with overlapping FFT windows
2. A single-pole transfer function correction is applied to account for the spatial averaging of the shear probe
3. Epsilon is estimated by fitting the observed spectrum to the theoretical Nasmyth spectrum in the inertial subrange
4. Unresolved high-wavenumber variance is accounted for using the integrated Nasmyth spectrum

## Python API

```python
from pyturb.processing import batch_compute_epsilon, bin_profiles
from pyturb.pfile import batch_convert_to_netcdf

# Convert P-files
batch_convert_to_netcdf('./path/to/raw_data/*.p', output_dir='./converted/')

# Compute epsilon
batch_compute_epsilon('./converted/*.nc', output_dir='./eps/', diss_len_sec=4.0)

# Bin profiles
bin_profiles('./eps/*.nc', output_file='binned.nc', bin_width=2.0)
```