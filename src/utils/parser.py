import argparse
import numpy as np
import netCDF4 as nc
import xarray as xr
import os
import json
import logging


def normalize(data, data_min: float, data_max: float):
    """
    Normalizes data to values from 0 to 1

    :param data: numpy array
    :param data_max: int
    :param data_min: int
    :return: numpy array
    """
    if data_min != data_max:
        return (data - data_min) / (data_max - data_min)
    else:
        return data


def data_steps(data, mmap, time_steps: int, delta: int):
    """
    Saves duplicate data as time steps to channels dim
    return_data.shape = (N, C, W, H)
    N - number of samples
    C - weather maps

    :param data: np.memmap
    :param mmap: np.memmap
    :param time_steps: int
    :param delta: int
    :return: numpy array
    """
    t_len = len(data) - (time_steps * delta)
    num_channels = len(data[0])

    for n in range(time_steps + 1):
        index = n * delta
        mmap[:, num_channels * n:num_channels * n + num_channels] = data[index:t_len + index]
        mmap.flush()

        logging.info(f'\nSaving sequences to memmap, step: {n + 1} out of total: {time_steps}')


parser = argparse.ArgumentParser(
    description='Data parser, it parses weather data from net4CDF format to numpy Memmap format'
)

parser.add_argument(
    '-i',
    '--input_file',
    type=str,
    nargs='+',
    help='Relative path to input data file or files',
    required=True
)

parser.add_argument(
    '-o',
    '--output_file',
    type=str,
    nargs='?',
    help='Relative path to the output file (If file does not exist then it will be created)',
    required=True
)

parser.add_argument(
    '--data_type',
    type=str,
    help='Datafile type, "netcdf" or "grib"',
    default="netcdf"
)

parser.add_argument(
    '--output_fn',
    type=str,
    help='Output file name',
    required=True
)

parser.add_argument(
    '-c',
    '--channels',
    type=str,
    nargs='+',
    help='All data file variables that need to be parsed to the new Memmap file',
    required=True
)

parser.add_argument(
    '-s',
    '--samples',
    type=int,
    default=-1,
    help='Amount of samples that are gonna be copied (default = -1, selects all samples from data file)',
)

parser.add_argument(
    '-t',
    '--time_steps',
    type=int,
    default=1,
    help='Regular: amount of time steps coded into C dim, RNN: sequence length (default = 1)',
)

parser.add_argument(
    '--delta',
    type=int,
    default=1,
    help='The time delta between each time step (default = 1)',
)

parser.add_argument(
    '--shuffle',
    type=lambda x: (str(x).lower() == 'true'),
    default=True,
    help='Shuffle data along the sample axis (default = True)',
)

parser.add_argument(
    '--normalize',
    type=lambda x: (str(x).lower() == 'true'),
    default=True,
    help='Normalize data or not (default = True)',
)

parser.add_argument(
    '--rnn',
    type=lambda x: (str(x).lower() == 'true'),
    default=False,
    help='Parse data for rnn model (default = False)',
)

args = parser.parse_args()

rootLogger = logging.getLogger()
logFormatter = logging.Formatter("%(asctime)s [%(process)d] [%(thread)d] [%(levelname)s]  %(message)s")
rootLogger.level = logging.DEBUG  # level

try:
    os.mkdir(args.output_file)
    logging.info(f'New directory created {args.output_file}')
except:
    logging.info(f'Directory already exists {args.output_file}')

open(f'{args.output_file}/log.txt', 'w')
fileHandler = logging.FileHandler(f'{args.output_file}/log.txt')
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

# Dataset variables
lon_versions = ['lon', 'longitude']
lat_versions = ['lat', 'latitude']
data_variables = args.channels

samples = args.samples

ds_list = []

if args.data_type == 'netcdf':
    ds_list = [nc.Dataset(i) for i in args.input_file]
elif args.data_type == 'grib':
    ds_list = [xr.open_dataset(i, engine="cfgrib") for i in args.input_file]

if not ds_list:
    raise ValueError('Wrong data type provided, available options: netcdf, grib')

validate_variables = data_variables.copy()
width = -1
height = -1
lon = None
lat = None

# ---------- Checks dataset data integrity ----------
for ds in ds_list:
    ds_var = ds.variables

    for n in range(len(lon_versions)):
        if lon_versions[n] in ds_var:
            if args.data_type == 'netcdf':
                val = ds_var[lon_versions[n]]
            elif args.data_type == 'grib':
                val = ds_var.get(lon_versions[n])

            if val is not None:
                if width == -1:
                    width = val.size

                elif width != val.size:
                    raise ValueError('W & H needs to be the same for all datasets')

                if lon is None:
                    lon = val[:]

                break

        elif n == len(lon_versions) - 1:
            raise ValueError('Lon variables were not found in the dataset')

    for n in range(len(lat_versions)):
        if lat_versions[n] in ds_var:
            if args.data_type == 'netcdf':
                val = ds_var[lat_versions[n]]
            elif args.data_type == 'grib':
                val = ds_var.get(lat_versions[n])

            if val is not None:
                if height == -1:
                    height = val.size

                elif height != val.size:
                    raise ValueError('W & H needs to be the same for all datasets')

                if lat is None:
                    lat = val[:]

                break

        elif n == len(lat_versions) - 1:
            raise ValueError('Lat variables were not found in the dataset')

    if width == -1 or height == -1:
        raise ValueError('Lon or lat variables were not found in the dataset')

    for v in data_variables:
        if v in ds_var and v in validate_variables:
            validate_variables.remove(v)

    time = ds_var['time'].size
    if samples != -1 and time < samples:
        raise ValueError('There are not enough samples in the dataset: ' + str(time))

if len(validate_variables) > 0:
    raise ValueError('All needed variables were not found in provided data files: ' + ', '.join(validate_variables))

data = np.array([])
min_max = {}
var_total_samples = np.zeros(len(data_variables))

logging.info('Collecting info about weather maps')
for idx, v in enumerate(data_variables):
    for ds in ds_list:
        ds_var = ds.variables

        if v in ds_var:
            ds_min = None
            ds_max = None
            ds_len = None

            if args.data_type == 'netcdf':
                ds_min = np.amin(ds_var[v])
                ds_max = np.amax(ds_var[v])
                ds_len = len(ds_var[v])
            elif args.data_type == 'grib':
                ds_min = np.amin(ds_var.get(v))
                ds_max = np.amax(ds_var.get(v))
                ds_len = len(ds_var.get(v))

            if ds_min is None or ds_max is None or ds_len is None:
                raise ValueError('variable ds_min, ds_max or ds_len is None')

            var_total_samples[idx] += ds_len

            if v not in min_max:
                min_max[v] = {'min': str(ds_min), 'max': str(ds_max)}
            else:
                if float(min_max[v]['min']) > ds_min:
                    min_max[v]['min'] = str(ds_min)

                if float(min_max[v]['max']) < ds_max:
                    min_max[v]['max'] = str(ds_max)

if samples != -1:
    total_samples = samples
elif (var_total_samples == var_total_samples[0]).all():
    total_samples = int(var_total_samples[0])
else:
    raise ValueError('Sample count for all variables is not the same')

# ---------- Calculates given data file size ----------
data_type = 'float32'
size = 0
for i in args.input_file:
    size += os.stat(i).st_size

# If size is over 100 GB then data is saved as float16 to save space
if size >= 10e8:
    data_type = 'float16'

# Sample count calculations
train_samples = int(total_samples * 0.8)
test_samples = int((total_samples - train_samples) * 0.5)
validate_samples = int(total_samples - train_samples - test_samples)

if args.rnn:
    channels = len(data_variables)
    seq_cut = args.delta * (args.time_steps - 1)

    # Mmap file shape definitions
    # (N, S, C, W, H)
    train_shape = (train_samples, channels, width, height)
    test_shape = (test_samples, channels, width, height)
    validate_shape = (validate_samples, len(data_variables), width, height)

else:
    channels = len(data_variables) * (args.time_steps + 1)
    seq_cut = args.time_steps * args.delta

    # Mmap file shape definitions
    # (N, C, W, H)
    train_shape = (train_samples - seq_cut, channels, width, height)
    test_shape = (test_samples - seq_cut, channels, width, height)
    validate_shape = (validate_samples, len(data_variables), width, height)

# Defines training and testing file names
data_train_filename = args.output_fn + '_train.mmap'
data_test_filename = args.output_fn + '_test.mmap'
data_validate_filename = args.output_fn + '_validate.mmap'

# ---------- Mmap creation ----------
mmap_train = np.memmap(filename=args.output_file + '/' + data_train_filename, dtype=data_type, mode='w+',
                       shape=train_shape)
mmap_test = np.memmap(filename=args.output_file + '/' + data_test_filename, dtype=data_type, mode='w+',
                      shape=test_shape)
mmap_validate = np.memmap(filename=args.output_file + '/' + data_validate_filename, dtype=data_type, mode='w+',
                          shape=validate_shape)

logging.info('Reading in all weather maps')

data = None
for idx_v, v in enumerate(data_variables):
    var_data = None

    for ds in ds_list:
        ds_var = ds.variables

        if v in ds_var:
            ds_data = None

            if args.data_type == 'netcdf':
                ds_data = ds_var[v][:]
            elif args.data_type == 'grib':
                ds_data = ds_var.get(v)[:]

            if samples != -1:
                ds_data = ds_data[:samples]

            corrupted_idxes = []
            for i in range(len(ds_data)):
                if np.isnan(ds_data[i].data).any() or np.isinf(ds_data[i].data).any() or None in ds_data[i].data:
                    corrupted_idxes.append(i)

            # NOTE this can delete maps in the middle of the data set which could mess with the time series order
            # Alternative solution would be to replace the corrupted values with something else
            if len(corrupted_idxes) > 0:
                logging.info(f'Corrupted indexes found for variable: {v}\n' +
                             f'Amount in total: {len(ds_data)}, deleted: {len(corrupted_idxes)}')
                ds_data = np.delete(ds_data, obj=corrupted_idxes, axis=0)

            # Normalization
            if args.normalize:
                ds_data = normalize(data=ds_data, data_min=float(min_max[v]['min']),
                                    data_max=float(min_max[v]['max']))

            ds_data = np.transpose(ds_data, axes=(0, 2, 1))
            ds_data = np.expand_dims(ds_data, axis=1)

            if var_data is None:
                var_data = ds_data
            else:
                var_data = np.concatenate((var_data, ds_data), axis=0)
            ds_data = None
    if data is None:
        data = var_data
    else:
        data = np.concatenate((data, var_data), axis=1)

logging.info('Saving sequences to memmap')

if args.rnn:
    mmap_train[:] = data[:train_samples]
    mmap_test[:] = data[train_samples:train_samples+test_samples]
    mmap_validate[:] = data[train_samples + test_samples:total_samples]
else:
    data_steps(data=data[:train_samples], mmap=mmap_train, time_steps=args.time_steps, delta=args.delta)
    data_steps(data=data[train_samples:train_samples + test_samples], mmap=mmap_test, time_steps=args.time_steps,
               delta=args.delta)
    mmap_validate[:] = data[train_samples + test_samples:total_samples]

# ---------- Push changes to disc ----------
mmap_train.flush()
mmap_test.flush()
mmap_validate.flush()

logging.info('Saving file metadata')

# ---------- Metadata definition ----------
ds_metadata = {
    'filename': data_train_filename,
    'shape': train_shape,
    'variables': data_variables,
    'var_max_min_values': min_max,
    'normalized': args.normalize,
    'shuffled': args.shuffle,
    'data_type': data_type,
    'time_steps': args.time_steps,
    'delta': args.delta,
}

# ---------- Metadata saving ----------
with open(args.output_file + '/' + args.output_fn + '_train.json', 'w') as j:
    json.dump(ds_metadata, j, indent=2)

ds_metadata['filename'] = data_test_filename
ds_metadata['shape'] = test_shape
with open(args.output_file + '/' + args.output_fn + '_test.json', 'w') as j:
    json.dump(ds_metadata, j, indent=2)

ds_metadata['filename'] = data_validate_filename
ds_metadata['shape'] = validate_shape
ds_metadata['shuffled'] = False
with open(args.output_file + '/' + args.output_fn + '_validate.json', 'w') as j:
    json.dump(ds_metadata, j, indent=2)

extra_metadata = {
    'shape': 2,
    'variables': ('lon', 'lat'),
    'data': [list(lon), list(lat)],
}

with open(args.output_file + '/' + args.output_fn + '_extra.json', 'w') as j:
    json.dump(extra_metadata, j, indent=2)

logging.info('Data parsing finished')
