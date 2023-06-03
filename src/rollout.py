from tqdm import tqdm
import argparse
import time
import numpy as np
import torch.utils.data
from torch import Tensor
from datetime import datetime
import csv

from utils import (
    metric_utils as utils,
    tensorboard_utils,
)
from utils.file_utils import FileUtils
from utils.csv_utils_2 import CsvUtils2
from utils.plot_utils import PlotUtils

import matplotlib.pyplot as plt


def rollout_key_idx(metric: str):
    key_idx = {f'{metric}_1_step': 1}

    if 72 <= args.delta * args.rollout:
        key_idx[f'{metric}_3_days'] = int(72 / args.delta)

    if 120 <= args.delta * args.rollout:
        key_idx[f'{metric}_5_days'] = int(120 / args.delta)

    return key_idx


USE_CUDA = torch.cuda.is_available()

parser = argparse.ArgumentParser(
    description='Script used for model rollout over some time range'
)

parser.add_argument('--sequence_name_orig', default='sequence', type=str)
parser.add_argument('--id', default=1, type=str)
parser.add_argument('--template', default='src/template_loc.sh', type=str)
parser.add_argument('--script', default='src/main.py', type=str)
parser.add_argument('--num_repeat',
                    help='how many times each set of parameters should be repeated for testing stability', default=1,
                    type=int)
parser.add_argument('--num_tasks_in_parallel', default=6, type=int)
parser.add_argument('--num_cuda_devices_per_task', default=1, type=int)
parser.add_argument('--is_single_task', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--is_force_start', default=True, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument('--rollout', default=1, type=int)
parser.add_argument('--x_time_steps', type=int)
parser.add_argument('--delta', default=1, type=int)
parser.add_argument('--batch_size', type=int)

parser.add_argument('--device', default='cuda', type=str)

parser.add_argument('--datasource', default='data_rollout', type=str)
parser.add_argument('--model_args', type=str, help='Relative path to run args', required=True)

parser.add_argument('--model_id', type=int, help='Id of the model', required=True)
parser.add_argument('--run_name', default=f'run_{time.time()}', type=str)
parser.add_argument('--sequence_name', default='seq_default_rollout', type=str)

parser.add_argument('--data_validate', type=str, help='Relative path to validation data',
                    default='../mmap_data/weather_data_validate.mmap')
parser.add_argument('--meta_validate', type=str, help='Relative path to validation metadata file',
                    default='../mmap_data/weather_data_validate.json')
parser.add_argument('--extra_data', type=str, help='Relative path to extra data file')

parser.add_argument('--map_row_count', default=3, type=int)
parser.add_argument('--inspect_amount', default=2, type=int)
parser.add_argument('--map_color', default='plasma', type=str)

args = parser.parse_args()

with open(args.model_args, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if int(row['id']) is args.model_id:
            model_args = row
            break

# If model path was wrong then an error is thrown
if model_args is None:
    raise ValueError('Error: Model id was not found in the provided csv file')

# Loads in the model and dataset from the /models and /data_sets directories
Model = getattr(__import__('models.' + model_args['model'], fromlist=['Model']), 'Model')
DataSet = getattr(__import__('data_sets.' + args.datasource, fromlist=['DataSet']), 'DataSet')

args.run_name += ('-' + datetime.utcnow().strftime(f'%y-%m-%d--%H-%M-%S'))

summary_writer = tensorboard_utils.CustomSummaryWriter(
    logdir=f'tensorboard/{args.sequence_name}/{args.run_name}'
)

# Creates directories for resulting CSV files
path_sequence = f'./results/{args.sequence_name}'
path_run = f'./results/{args.sequence_name}/{args.run_name}'
path_artificats = f'./artifacts/{args.sequence_name}/{args.run_name}'
FileUtils.createDir(path_run)
FileUtils.createDir(path_artificats)
FileUtils.writeJSON(f'{path_run}/args.json', args.__dict__)

CsvUtils2.create_global(path_sequence)
CsvUtils2.create_local(path_sequence, args.run_name)

channels = model_args['channel'].split('.')

args.forecast_channel = model_args['forecast_channel'] if 'forecast_channel' in model_args else channels[0]
args.channel = channels
args.div_groupnorm = int(model_args['div_groupnorm'])

args.rnn_layer_count = None
if args.rollout is None and 'rollout' in model_args:
    args.rollout = int(model_args['rollout'])
if args.x_time_steps is None and 'x_time_steps' in model_args:
    args.x_time_steps = int(model_args['x_time_steps'])
if args.extra_data is None and 'extra_data' in model_args:
    args.extra_data = model_args['extra_data']
if 'rnn_layer_count' in model_args:
    args.rnn_layer_count = int(model_args['rnn_layer_count'])
if 'delta' in model_args:
    args.delta = int(model_args['delta'])
if 'width' in model_args and 'height' in model_args:
    args.width = int(model_args['width'])
    args.height = int(model_args['height'])

args.ds_type = 'validate'

# Data loaders for validation data
data_loader_validate = torch.utils.data.DataLoader(
    dataset=DataSet(
        data_path=args.data_validate,
        metadata_path=args.meta_validate,
        args=args,
    ),
    batch_size=args.batch_size if args.batch_size is not None else int(model_args['batch_size']),
    drop_last=True,
    shuffle=False,
)

args.ds_type = None

model = Model(
    args=args,
)

# Loads in a saved model
model.load_state_dict(torch.load(model_args['model_dir']))

# Puts model to device, if 'cuda' then the GPU is used
model = model.to(args.device)

# If more than one GPU then model has the option to run on many GPUs
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, dim=0)

model.eval()

metric_list = ['rmse', 'nrmse', 'r2score', 'weighted_rmse', 'relative_r2score']

metrics_rollout = {}
# Creates array for chosen rollout metrics
if args.rollout is not None and args.delta is not None:
    for metric in metric_list:
        keys = rollout_key_idx(metric)

        for k in keys.keys():
            metrics_rollout[k] = []

args.forecast_idx = 0
if data_loader_validate.dataset.channel_y_idx_in_x is not None:
    args.forecast_idx = data_loader_validate.dataset.channel_y_idx_in_x

for x, y in tqdm(data_loader_validate):
    if args.device == 'cuda':
        x = x.cuda()
        y = y.cuda()

    y_prim = None
    if len(x.shape) == 5:  # (B, S, C, W, H)
        # RNN UNet model
        if x.shape[2] > 1:  # If C dim is > 1
            x_extra_channels = list(np.arange(0, x.shape[2]))
            x_extra_channels.pop(args.forecast_idx)

            # Extra channels are encoded in the hidden state
            hidden = model.init_hidden(x[:, 0, x_extra_channels])
            y_prim, hidden = model.forward(x[:, :, args.forecast_idx].unsqueeze(dim=2), hidden)
        else:
            # Single channel is used for forecasting
            hidden = model.init_hidden(args.batch_size)
            y_prim, hidden = model.forward(x, hidden)

        y_prim = y_prim[:, -1].unsqueeze(1)

        for _ in range(args.rollout - 1):
            input = y_prim[:, -1].unsqueeze(1)
            y_prim_step, hidden = model.forward(input, hidden)
            y_prim = torch.cat([y_prim, y_prim_step], dim=1)
    else:
        y_prim = model.forward(x)

        for _ in range(args.rollout - 1):
            input = y_prim[:, -1].unsqueeze(1)
            y_prim_step = model.forward(input)
            y_prim = torch.cat([y_prim, y_prim_step], dim=1)

    # Calculates all metrics from the metric list
    if 'rmse' in metric_list:
        calc_y = y
        calc_y_prim = y_prim

        if data_loader_validate.dataset.normalize:
            min = float(data_loader_validate.dataset.min)
            max = float(data_loader_validate.dataset.max)
            calc_y = (calc_y * (max - min)) + min
            calc_y_prim = (calc_y_prim * (max - min)) + min

        key_idx = rollout_key_idx('rmse')

        # if args.rollout is not None and args.delta is not None:
        for key, idx in key_idx.items():
            metric_rmse = utils.rmse(calc_y[:, :idx], calc_y_prim[:, :idx])
            metrics_rollout[key].append(metric_rmse.item())

    if 'nrmse' in metric_list:
        key_idx = rollout_key_idx('nrmse')

        for key, idx in key_idx.items():
            metric_nrmse = utils.nrmse(y[:, :idx], y_prim[:, :idx], not data_loader_validate.dataset.normalize,
                                       float(data_loader_validate.dataset.min),
                                       float(data_loader_validate.dataset.max))
            metrics_rollout[key].append(metric_nrmse.item())

    if 'r2score' in metric_list:
        key_idx = rollout_key_idx('r2score')

        for key, idx in key_idx.items():
            metric_r2score = utils.r2score(y[:, :idx], y_prim[:, :idx])
            metrics_rollout[key].append(metric_r2score.item())

    if 'weighted_rmse' in metric_list:
        calc_y = y
        calc_y_prim = y_prim

        # If data is normalized then it is needed to denormalize it
        if data_loader_validate.dataset.normalize:
            min = float(data_loader_validate.dataset.min)
            max = float(data_loader_validate.dataset.max)
            calc_y = (calc_y * (max - min)) + min
            calc_y_prim = (calc_y_prim * (max - min)) + min

        lat = torch.FloatTensor(data_loader_validate.dataset.lat)
        if args.device == 'cuda':
            lat = lat.cuda()

        key_idx = rollout_key_idx('weighted_rmse')

        for key, idx in key_idx.items():
            metric_weighted_rmse = utils.weighted_rmse(calc_y[:, :idx], calc_y_prim[:, :idx], lat)
            metrics_rollout[key].append(metric_weighted_rmse.item())

    if 'relative_r2score' in metric_list:
        key_idx = rollout_key_idx('relative_r2score')

        for key, idx in key_idx.items():
            if len(x.shape) == 5:  # (B, S, C, W, H)
                single_x = x[:, -1]
                single_y_prim = y_prim[:, idx - 1]
                single_y = y[:, idx - 1]
            else:  # (B, C, W, H)
                if x.shape[1] > 1:  # If C is > 1
                    single_x = x[:, args.forecast_idx].unsqueeze(dim=1)
                    single_y_prim = y_prim[:, idx - 1]
                    single_y = y[:, idx - 1]
                else:
                    single_x = x
                    single_y_prim = y_prim[:, idx - 1]
                    single_y = y[:, idx - 1]

            metric_relative_r2score = utils.relative_r2score(single_x, single_y, single_y_prim)
            if isinstance(metric_relative_r2score, Tensor):
                metric_relative_r2score = metric_relative_r2score.item()

            metrics_rollout[key].append(metric_relative_r2score)

    # Variables are brought down to cpu
    y_prim = y_prim.cpu()
    x = x.cpu()
    y = y.cpu()

    # This ensures that the last batch will be usable outside of cycle
    np_x = x.data.numpy()
    np_y = y.data.numpy()
    np_y_prim = y_prim.data.numpy()

# This will store every metric mean over all forecasts
metric_means = {}

if args.rollout is not None and args.delta is not None:
    for metric in metric_list:
        keys = rollout_key_idx(metric)

        for k in keys.keys():
            metric_means[f'{k}'] = []

# Calculates mean for every metric
# {'rmse_1_step': 290.7}
for key in metric_means.keys():
    metric_means[key] = np.mean(metrics_rollout[key])

amount_of_samples = args.map_row_count * args.inspect_amount
# Reshapes and selects data that will be plotted
if len(np_x.shape) == 5:
    # (B, S, C, W, H) => (B, S, H, W)
    np_x = np.transpose(np_x, (0, 1, 2, 4, 3))[:amount_of_samples, :, args.forecast_idx]
    np_y_prim = np.transpose(np_y_prim, (0, 1, 2, 4, 3))[:amount_of_samples, :, 0]
    np_y = np.transpose(np_y, (0, 1, 2, 4, 3))[:amount_of_samples, :, 0]
else:
    # (B, C, W, H) => (B, S, H, W)
    np_x = np.transpose(np_x, (0, 1, 3, 2))[:amount_of_samples, [args.forecast_idx]]
    np_y_prim = np.transpose(np_y_prim, (0, 1, 3, 2))[:amount_of_samples]
    np_y = np.transpose(np_y, (0, 1, 3, 2))[:amount_of_samples]

# Min and max values of the forecast_channel data
data_min = 0
data_max = 1
if not data_loader_validate.dataset.normalize:
    data_min = float(data_loader_validate.dataset.min)
    data_max = float(data_loader_validate.dataset.max)

# Log makes the difference between pixels more visible
eps = 1e-8
np_x_log = - np.log(np_x + eps)
np_y_prim_log = - np.log(np_y_prim + eps)
np_y_log = - np.log(np_y + eps)

# Calculates metrics for each step and creates pictures for each step to save to Tensorboard
for step in range(1, args.rollout + 1):
    # Calculates metric mean for step and prints it to console
    metric_str = []

    summary_writer.add_hparams(
        hparam_dict=args.__dict__,
        metric_dict=metric_means,
        global_step=step,
    )

    CsvUtils2.add_hparams(
        path_sequence=path_sequence,
        run_name=args.run_name,
        args_dict=args.__dict__,
        metrics_dict=metric_means,
        global_step=step,
    )

    for idx, i in enumerate(range(0, len(np_x), args.map_row_count)):
        fig1 = PlotUtils.plot_regular_grid(
            row_count=args.map_row_count,
            x=np_x[i:i + args.map_row_count, 0],
            y_prim=np_y_prim[i:i + args.map_row_count, step - 1],
            y=np_y[i:i + args.map_row_count, step - 1],
            data_min=data_min,
            data_max=data_max,
            title_ax1='x',
            title_ax2='y_prim',
            title_ax3='y',
            map_color=args.map_color,
        )

        fig2 = PlotUtils.plot_regular_grid(
            row_count=args.map_row_count,
            x=np_x_log[i:i + args.map_row_count, 0],
            y_prim=np_y_prim_log[i:i + args.map_row_count, step - 1],
            y=np_y_log[i:i + args.map_row_count, step - 1],
            data_min=data_min,
            data_max=data_max,
            title_ax1='x',
            title_ax2='y_prim',
            title_ax3='y',
            map_color=args.map_color,
        )

        fig1.suptitle(f't{step} ({step * args.delta} hours)', fontsize=24)
        fig2.suptitle(f't{step} ({step * args.delta} hours)', fontsize=24)

        plt.tight_layout(pad=0.5)

        summary_writer.add_figure(
            tag=f'inspect_{idx}',
            figure=fig1,
            global_step=step,
        )

        summary_writer.add_figure(
            tag=f'inspect_log_{idx}',
            figure=fig2,
            global_step=step,
        )

    plt.clf()
    summary_writer.flush()
