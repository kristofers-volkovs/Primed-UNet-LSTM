import torch
import json
import numpy as np

import torch.utils.data


class DataSet(torch.utils.data.Dataset):
    def __init__(self, data_path: str, metadata_path: str, args):

        with open(metadata_path, 'r') as j:
            obj = json.load(j)
            self.normalize = obj['normalized']

            data_type = obj['data_type']
            data_variables = obj['variables']

            if args.forecast_channel not in data_variables:
                raise ValueError(
                    f'Provided channel variable was not found in the provided data, channel: {args.forecast_channel}'
                )

            self.min = obj['var_max_min_values'][args.forecast_channel]['min']
            self.max = obj['var_max_min_values'][args.forecast_channel]['max']

            self.variable_len = len(data_variables)

            self.y_channel = data_variables.index(args.forecast_channel)
            self.x_channel = []

            # Index of forecast channel in x_channel array
            self.channel_y_idx_in_x = None

            # Collects all used channel idxes in one array
            for c in args.channel:
                self.x_channel.append(data_variables.index(c))

                # If forecast channel abbreviation is the same as channel abbreviation the its idx is saved
                if c == args.forecast_channel:
                    self.channel_y_idx_in_x = len(self.x_channel) - 1

            self.rollout = args.rollout
            self.delta = args.delta
            self.seq_cut = args.delta * args.rollout

            self.data = np.memmap(filename=data_path, dtype=data_type, mode='r', shape=tuple(obj['shape']))

        with open(args.extra_data, 'r') as j:
            obj = json.load(j)

            lon_idx = obj['variables'].index('lon')
            lat_idx = obj['variables'].index('lat')
            self.lon = obj['data'][lon_idx]
            self.lat = obj['data'][lat_idx]

    def __len__(self):
        return len(self.data) - self.seq_cut

    def __getitem__(self, idx):
        if self.variable_len == len(self.x_channel):
            x = torch.FloatTensor(self.data[idx].copy())
        else:
            x = torch.FloatTensor(self.data[idx, self.x_channel].copy())

        y_end_idx = idx + self.delta + self.rollout * self.delta
        y_idx_arange = np.arange(idx + self.delta, y_end_idx, self.delta)
        y = torch.FloatTensor(self.data[y_idx_arange, self.y_channel].copy())

        return x, y
