import argparse
import time
import logging
from datetime import datetime
import numpy as np
import torch.utils.data
from torch import Tensor
from tqdm import tqdm
import torch_optimizer as optim
from utils import (
    metric_utils as utils,
    tensorboard_utils,
)
from utils.file_utils import FileUtils
from utils.csv_utils_2 import CsvUtils2
from utils.plot_utils import PlotUtils
# from apex import amp

from functools import reduce

import matplotlib.pyplot as plt


def rollout_key_idx(metric: str):
    key_idx = {f'{metric}_rollout_1_time_step': 1}

    if 72 <= args.delta * args.rollout:
        key_idx[f'{metric}_rollout_3_days'] = int(72 / args.delta)

    if 120 <= args.delta * args.rollout:
        key_idx[f'{metric}_rollout_5_days'] = int(120 / args.delta)

    return key_idx


USE_CUDA = torch.cuda.is_available()

parser = argparse.ArgumentParser(
    description='Training script used for model training'
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

parser.add_argument('--model', default='model_unet', type=str)
parser.add_argument('--datasource', default='data_unet', type=str)

parser.add_argument('--run_name', default=f'run_{time.time()}', type=str)
parser.add_argument('--sequence_name', default='seq_default', type=str)

parser.add_argument('--learning_rate', default=1e-3, type=float)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--div_groupnorm', default=2, type=int)
parser.add_argument('--rnn_layer_count', type=int)
parser.add_argument('--rollout', type=int)
parser.add_argument('--x_time_steps', type=int)
parser.add_argument('--delta', type=int)

parser.add_argument('--device', default='cuda', type=str)

parser.add_argument('--loss_fn', default='nrmse', type=str)
parser.add_argument('--gamma', default=0.5, type=float)

parser.add_argument('--data_train', type=str, help='Relative path to training data',
                    default='../mmap_data/weather_data_train.mmap')
parser.add_argument('--meta_train', type=str, help='Relative path to training metadata file',
                    default='../mmap_data/weather_data_train.json')
parser.add_argument('--data_test', type=str, help='Relative path to testing data',
                    default='../mmap_data/weather_data_test.mmap')
parser.add_argument('--meta_test', type=str, help='Relative path to testing metadata file',
                    default='../mmap_data/weather_data_test.json')
parser.add_argument('--extra_data', type=str, help='Relative path to extra data file',
                    default='../mmap_data/weather_data_extra.json')

parser.add_argument('--channel', type=str, nargs='+', help='Data variable name (Default: t)', default='t')
parser.add_argument('--forecast_channel', type=str, help='Data variable to be forecasted (Default: t)', default='t')

parser.add_argument('--best_metric', default='rmse', type=str)
parser.add_argument('--early_stopping_patience', default=5, type=int)
parser.add_argument('--early_stopping_delta_percent', default=1e-2, type=float)
parser.add_argument('--early_stopping_param_coef', default=-1.0, type=float)

parser.add_argument('--map_row_count', default=3, type=int)
parser.add_argument('--inspect_amount', default=2, type=int)
parser.add_argument('--map_color', default='plasma', type=str)

args = parser.parse_args()

# Loads in the model and dataset from the /models and /data_sets directories
Model = getattr(__import__('models.' + args.model, fromlist=['Model']), 'Model')
DataSet = getattr(__import__('data_sets.' + args.datasource, fromlist=['DataSet']), 'DataSet')

args.run_name += ('-' + datetime.utcnow().strftime(f'%y-%m-%d--%H-%M-%S'))
args.model_dir = f'./results/{args.sequence_name}/{args.run_name}-id{args.id}.pt'

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

args.ds_type = 'train'

# Data loaders for train and test data
data_loader_train = torch.utils.data.DataLoader(
    dataset=DataSet(
        data_path=args.data_train,
        metadata_path=args.meta_train,
        args=args,
    ),
    batch_size=args.batch_size,
    drop_last=True,
    shuffle=True,
)
args.ds_type = 'test'

data_loader_test = torch.utils.data.DataLoader(
    dataset=DataSet(
        data_path=args.data_test,
        metadata_path=args.meta_test,
        args=args,
    ),
    batch_size=args.batch_size,
    drop_last=True,
    shuffle=False,
)
args.ds_type = None

args.forecast_idx = 0

if data_loader_train.dataset.channel_y_idx_in_x is not None:
    args.forecast_idx = data_loader_train.dataset.channel_y_idx_in_x

model = Model(
    args=args,
)
optimizer = optim.RAdam(params=model.parameters(), lr=args.learning_rate)

# If 'cuda' then GPU is used
model = model.to(args.device)

# if args.device == 'cuda':
#     opt_level = 'O1'
#     model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

# If more than one GPU available then training is run on many GPUs
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, dim=0)

metric_list = ['loss', 'rmse', 'nrmse', 'r2score', 'weighted_rmse', 'relative_r2score']

# Creates array where all calculated metrics for all epochs will be stored
metrics = {}
for stage in ['train', 'test']:
    for metric in metric_list:
        metrics[f'{stage}_{metric}'] = []

# Creates array for chosen rollout metrics
if args.rollout is not None and args.delta is not None:
    rollout_metric_list = ['rmse', 'nrmse', 'weighted_rmse']
    for metric in rollout_metric_list:
        keys = rollout_key_idx(metric)

        for k in keys.keys():
            metrics[k] = []

args.channel = reduce(lambda arg1, arg2: arg1 + '.' + arg2, args.channel)
best_loss = float('Inf')
best_nrmse = float('Inf')

# Variable definition for early stopping
prev_best_metric = 0
early_stopping_patience = 0
for epoch in range(1, args.epochs):
    metric_epoch_result = {}

    # Goes through the data_loaders, trains the model, calculates and saves metrics
    for data_loader in [data_loader_train, data_loader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        torch.set_grad_enabled(True)
        model = model.train()

        # When test data then model is being evaluated
        if data_loader == data_loader_test:
            stage = 'test'
            # Small optimization for speed
            torch.set_grad_enabled(False)
            model = model.eval()

        for x, y in tqdm(data_loader):
            if args.device == 'cuda':
                x = x.cuda()
                y = y.cuda()

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

                # Rollout is made only for test run
                if data_loader == data_loader_test:
                    y_prim = y_prim[:, -1].unsqueeze(1)
                    for _ in range(args.rollout - 1):
                        input = y_prim[:, -1].unsqueeze(1)
                        y_prim_step, hidden = model.forward(input, hidden)
                        y_prim = torch.cat([y_prim, y_prim_step], dim=1)

            else:  # (B, C, W, H)
                # Simple UNet model
                y_prim = model.forward(x)

            # Calculates loss depending on which one was passed in the args
            loss = None
            if args.loss_fn == 'mae':
                loss = utils.mae(y, y_prim)

            elif args.loss_fn == 'mse':
                loss = utils.mse(y, y_prim)

            elif args.loss_fn == 'r2score':
                loss = utils.r2score(y, y_prim)

            elif args.loss_fn == 'huber':
                loss = utils.huber(y, y_prim, args.gamma)

            elif args.loss_fn == 'rmse':
                loss = utils.rmse(y, y_prim)

            elif args.loss_fn == 'nrmse':
                loss = utils.nrmse(y, y_prim, not data_loader.dataset.normalize, float(data_loader.dataset.min),
                                   float(data_loader.dataset.max))

            elif args.loss_fn == 'kl':
                loss = utils.kl(y, y_prim, not data_loader.dataset.normalize, float(data_loader.dataset.min),
                                float(data_loader.dataset.max), 1e-8)

            elif args.loss_fn == 'focalKL':
                loss = utils.focalKL(y, y_prim, not data_loader.dataset.normalize, float(data_loader.dataset.min),
                                     float(data_loader.dataset.max), args.gamma, 1e-8)

            # If loss was inputted wrong then an error is thrown
            if loss is None:
                raise ValueError('Error: invalid loss_fn was provided')

            metrics_epoch[f'{stage}_loss'].append(loss.item())

            # Calculates all metrics from the metric list
            if 'rmse' in metric_list:
                calc_y = y
                calc_y_prim = y_prim

                # If data is normalized then it is needed to denormalize it
                if data_loader.dataset.normalize:
                    min = float(data_loader.dataset.min)
                    max = float(data_loader.dataset.max)
                    calc_y = (calc_y * (max - min)) + min
                    calc_y_prim = (calc_y_prim * (max - min)) + min

                metric_rmse = utils.rmse(calc_y, calc_y_prim)
                metrics_epoch[f'{stage}_rmse'].append(metric_rmse.item())

                # When test cycle, calculates different rmse time step values
                if data_loader == data_loader_test and args.rollout is not None and args.delta is not None:
                    key_idx = rollout_key_idx('rmse')

                    for key, idx in key_idx.items():
                        metric_rmse = utils.rmse(calc_y[:, :idx], calc_y_prim[:, :idx])
                        metrics_epoch[key].append(metric_rmse.item())

            if 'nrmse' in metric_list:
                metric_nrmse = utils.nrmse(y, y_prim, not data_loader.dataset.normalize, float(data_loader.dataset.min),
                                           float(data_loader.dataset.max))
                metrics_epoch[f'{stage}_nrmse'].append(metric_nrmse.item())

                # When test cycle, calculates different nrmse time step values
                if data_loader == data_loader_test and args.rollout is not None and args.delta is not None:
                    key_idx = rollout_key_idx('nrmse')

                    for key, idx in key_idx.items():
                        metric_rmse = utils.nrmse(y[:, :idx], y_prim[:, :idx], not data_loader.dataset.normalize,
                                                  float(data_loader.dataset.min), float(data_loader.dataset.max))
                        metrics_epoch[key].append(metric_rmse.item())

            if 'r2score' in metric_list:
                metric_r2 = utils.r2score(y, y_prim)
                metrics_epoch[f'{stage}_r2score'].append(metric_r2.item())

            if 'weighted_rmse' in metric_list:
                calc_y = y
                calc_y_prim = y_prim

                # If data is normalized then it is needed to denormalize it
                if data_loader.dataset.normalize:
                    min = float(data_loader.dataset.min)
                    max = float(data_loader.dataset.max)
                    calc_y = (calc_y * (max - min)) + min
                    calc_y_prim = (calc_y_prim * (max - min)) + min

                lat = torch.FloatTensor(data_loader.dataset.lat)
                if args.device == 'cuda':
                    lat = lat.cuda()

                metric_weighted_rmse = utils.weighted_rmse(calc_y, calc_y_prim, lat)
                metrics_epoch[f'{stage}_weighted_rmse'].append(metric_weighted_rmse.item())

                # When test cycle, calculates different weighted rmse time step values
                if data_loader == data_loader_test and args.rollout is not None and args.delta is not None:
                    key_idx = rollout_key_idx('weighted_rmse')

                    for key, idx in key_idx.items():
                        metric_rmse = utils.weighted_rmse(calc_y[:, :idx], calc_y_prim[:, :idx], lat)
                        metrics_epoch[key].append(metric_rmse.item())

            if 'relative_r2score' in metric_list:
                if len(x.shape) == 5:  # (B, S, C, W, H)
                    single_x = x[:, -1]
                    single_y_prim = y_prim[:, -1]
                    single_y = y[:, -1]
                else:  # (B, C, W, H)
                    if x.shape[1] > 1:
                        single_x = x[:, args.forecast_idx].unsqueeze(dim=1)
                        single_y_prim = y_prim
                        single_y = y
                    else:
                        single_x = x
                        single_y_prim = y_prim
                        single_y = y

                metric_relative_r2score = utils.relative_r2score(single_x, single_y, single_y_prim)
                if isinstance(metric_relative_r2score, Tensor):
                    metric_relative_r2score = metric_relative_r2score.item()

                metrics_epoch[f'{stage}_relative_r2score'].append(metric_relative_r2score)

            # Backpropagation for training, updates weights and biases
            if data_loader == data_loader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Variables are brought down to cpu
            loss = loss.cpu()
            y_prim = y_prim.cpu()
            x = x.cpu()
            y = y.cpu()

            # This ensures that the last batch will be usable outside of cycle
            np_x = x.data.numpy()
            np_y = y.data.numpy()
            np_y_prim = y_prim.data.numpy()

        # Calculates all metric mean and prints it to console
        metrics_strs = []
        for key in metrics_epoch.keys():
            if stage in key:
                value = np.mean(metrics_epoch[key])
                metrics[key].append(value)
                metrics_strs.append(f'{key}: {round(value, 2)}')

        if data_loader is data_loader_test and args.rollout is not None and args.delta is not None:
            for metric in rollout_metric_list:
                keys = rollout_key_idx(metric)

                for k in keys.keys():
                    value = np.mean(metrics_epoch[k])
                    metrics[k].append(value)
                    metrics_strs.append(f'{k}: {round(value, 2)}')

        print(f'epoch: {epoch} {" ".join(metrics_strs)}')

        # Saves current epoch training metrics
        if data_loader is data_loader_train:
            for metric in metric_list:
                metric_epoch_result[f'train_{metric}'] = metrics[f'train_{metric}'][-1]

        # Saves current epoch test metrics & saves everything to CSV and Tensorboard
        if data_loader is data_loader_test:
            for metric in metric_list:
                metric_epoch_result[f'test_{metric}'] = metrics[f'test_{metric}'][-1]

            if best_loss > metrics['test_loss'][-1]:
                best_loss = metrics['test_loss'][-1]

            if best_nrmse > metrics['test_nrmse'][-1]:
                best_nrmse = metrics['test_nrmse'][-1]

            metric_epoch_result['best_loss'] = best_loss
            metric_epoch_result['best_nrmse'] = best_nrmse

            # Saves different time step values for chosen rollout metrics
            if args.rollout is not None and args.delta is not None:
                for metric in rollout_metric_list:
                    keys = rollout_key_idx(metric)

                    for k in keys.keys():
                        metric_epoch_result[k] = metrics[k][-1]

            summary_writer.add_hparams(
                hparam_dict=args.__dict__,
                metric_dict=metric_epoch_result,
                global_step=epoch,
            )

            CsvUtils2.add_hparams(
                path_sequence=path_sequence,
                run_name=args.run_name,
                args_dict=args.__dict__,
                metrics_dict=metric_epoch_result,
                global_step=epoch,
            )

            metric_epoch_result = {}

            amount_of_samples = args.map_row_count * args.inspect_amount
            # Reshapes and selects data that will be plotted
            if len(np_x.shape) == 5:
                # (B, S, C, W, H) => (B, H, W)
                np_x = np.transpose(np_x, (0, 1, 2, 4, 3))[:amount_of_samples, -1, args.forecast_idx]
                np_y_prim = np.transpose(np_y_prim, (0, 1, 2, 4, 3))[:amount_of_samples, -1, 0]
                np_y = np.transpose(np_y, (0, 1, 2, 4, 3))[:amount_of_samples, -1, 0]
            else:
                # (B, C, W, H) => (B, H, W)
                np_x = np.transpose(np_x, (0, 1, 3, 2))[:amount_of_samples, args.forecast_idx]
                np_y_prim = np.transpose(np_y_prim, (0, 1, 3, 2))[:amount_of_samples, 0]
                np_y = np.transpose(np_y, (0, 1, 3, 2))[:amount_of_samples, 0]

            # Min and max values of the forecast_channel data
            data_min = 0
            data_max = 1
            if not data_loader.dataset.normalize:
                data_min = float(data_loader.dataset.min)
                data_max = float(data_loader.dataset.max)

            # Log makes the difference between pixels more visible
            eps = 1e-8
            np_x_log = - np.log(np_x + eps)
            np_y_prim_log = - np.log(np_y_prim + eps)
            np_y_log = - np.log(np_y + eps)

            for idx, i in enumerate(range(0, len(np_x), args.map_row_count)):
                fig1 = PlotUtils.plot_regular_grid(
                    row_count=args.map_row_count,
                    x=np_x[i:i + args.map_row_count],
                    y_prim=np_y_prim[i:i + args.map_row_count],
                    y=np_y[i:i + args.map_row_count],
                    data_min=data_min,
                    data_max=data_max,
                    title_ax1='x',
                    title_ax2='y_prim',
                    title_ax3='y',
                    map_color=args.map_color,
                )

                fig2 = PlotUtils.plot_regular_grid(
                    row_count=args.map_row_count,
                    x=np_x_log[i:i + args.map_row_count],
                    y_prim=np_y_prim_log[i:i + args.map_row_count],
                    y=np_y_log[i:i + args.map_row_count],
                    data_min=data_min,
                    data_max=data_max,
                    title_ax1='x',
                    title_ax2='y_prim',
                    title_ax3='y',
                    map_color=args.map_color,
                )

                plt.tight_layout(pad=0.5)

                summary_writer.add_figure(
                    tag=f'inspect_{idx}',
                    figure=fig1,
                    global_step=epoch,
                )

                summary_writer.add_figure(
                    tag=f'inspect_log_{idx}',
                    figure=fig2,
                    global_step=epoch,
                )

            plt.clf()
            summary_writer.flush()

    # Early stopping
    percent_improvement = 0
    if epoch > 1:
        best_metric_val = 0
        # Finds the best metric from test metrics and takes a mean of it
        for m in metrics_epoch.keys():
            metric_split = m.split('_')
            if metric_split[0] == 'test' and metric_split[1] == args.best_metric:
                best_metric_val = np.mean(metrics_epoch[m])
                break

        if prev_best_metric != 0:
            # If best metric is nan or inf then model is saved and script stopped
            if np.isnan(best_metric_val).any() or np.isinf(best_metric_val).any():
                logging.info('loss isnan break')
                torch.save(model.state_dict(), args.model_dir)
                break

            percent_improvement = args.early_stopping_param_coef * (
                    best_metric_val - prev_best_metric) / prev_best_metric
            # Improvement percent is reset if it's nan
            if np.isnan(percent_improvement).any():
                percent_improvement = 0

            if best_metric_val >= 0:
                # If improvement is worse than delta percent then patience += 1
                if args.early_stopping_delta_percent > percent_improvement:
                    early_stopping_patience += 1
                else:
                    early_stopping_patience = 0

        # If patience is over args patience then model is saved and script is stopped
        if early_stopping_patience > args.early_stopping_patience:
            logging.info('early_stopping_patience break')
            torch.save(model.state_dict(), args.model_dir)
            break

        if epoch == args.epochs - 1:
            logging.info('last epoch save')
            torch.save(model.state_dict(), args.model_dir)

        prev_best_metric = best_metric_val

summary_writer.close()
