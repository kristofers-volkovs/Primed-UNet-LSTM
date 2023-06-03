# **Primed-UNet-LSTM**

## **Tech stack**

    - Python
    - PyTorch
    - Conda
    - Docker

---

## **Automatic setup prerequisites**

    - Docker
    - Bash

- Download Docker: [link](https://docs.docker.com/get-started/)
- Setup Docker user (for Linux): [link](https://docs.docker.com/engine/install/linux-postinstall/)
- Setup NVIDIA container toolkit: [link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)


---

## **Manual setup prerequisites**

    - Conda

Environment for parsing files

```sh
conda env create -f ./conda-env/parsers_env.yml
```

Environment to run the model training script

```sh
conda env create -f ./conda-env/torch_env.yml
```

---

## **Why Docker**

Docker is not used for deployment purposes for this project. The main reason
for Docker is that it takes out the tedious work of seting up an environtment
to run various scripts. Instead the setup is already defined in the Dockerfile
and all you need to do is compile it to an image and then use it to create a container.

In this project Docker is mainly used as an environment for executing various scripts
like scripts for data fetching and parsing and starting the model training process.
To achieve this when docker image is run it creates a container ( --rm flag removes
container when it stops ) and links this projects directory as its volume which
allows docker to see and execute all the files in the projects directory.

**<span style="color:red">Disclaimer:</span>** torch docker image has CUDA 11.0 configured
so to run the model on GPU it is needed to have a pc that is compatible with CUDA 11.0


Parser docker image doesn't use CUDA so there shouldn't be any problems with using it

---

## **Automatic shell scripts**

To make the whole process of executing various different scripts, shell scripts were
made to partially automate the whole process

This will download 6 different variables and 2 months (1416 samples) of ERA5 weather
maps from the CDS API. For the command to work it is needed to add th secret code, code
can be gotten here: [link](https://cds.climate.copernicus.eu/api-how-to)

```sh
bash scripts/fetch.sh -p docker/parser_docker \
-c '<Secret code>' \
--mode single \
--variable 2m_temperature 10m_u_component_of_wind \
10m_v_component_of_wind total_cloud_cover total_precipitation \
toa_incident_solar_radiation \
--level_type single \
--output_dir data/1979_t2m_u10_v10_tcc_tp_tisr \
--years 1979 \
--month 01 02
```

To download variables used in WeatherBench research paper: T850 and Z500

```sh
bash scripts/fetch.sh -p docker/parser_docker \
    -c '<Secret code>' \
    --mode single \
    --variable geopotential \
    --level_type pressure \
    --pressure_level 500 \
    --output_dir data/1979_t2m_u10_v10_tcc_tp_tisr_z_t \
    --custom_fn era5_z500 \
    --years 1979 \
    --month 01 02
```

```sh
bash scripts/fetch.sh -p docker/parser_docker \
    -c '<Secret code>' \
    --mode single \
    --variable temperature \
    --level_type pressure \
    --pressure_level 850 \
    --output_dir data/1979_t2m_u10_v10_tcc_tp_tisr_z_t \
    --custom_fn era5_t850 \
    --years 1979 \
    --month 01 02
```

(Optional) - to compare the results to WeatherBench results it is needed to change
the data resolution to one that WeatherBench model used

```sh
bash scripts/parse_wb.sh -p docker/parser_docker \
--input_dir data/1979_t2m_u10_v10_tcc_tp_tisr  \
--output_dir data/regrid_1979_t2m_u10_v10_tcc_tp_tisr_2.8125deg \
--ddeg_out 2.8125
```

To be able to use big amounts of data it is needed to parse them to memmap format,
this data is used by the model

```sh
bash scripts/parse_mmap.sh -p docker/parser_docker \
--input_dir data/regrid_1979_t2m_u10_v10_tcc_tp_tisr_2.8125deg \
--output_dir mmap_data \
--channels t2m u10 v10 tcc
```

Compiles docker image and starts the training process of the model

```sh
bash scripts/start_train.sh -p docker/torch_docker \
--d_train mmap_data/weather_data_train.mmap \
--m_train mmap_data/weather_data_train.json \
--d_test mmap_data/weather_data_test.mmap \
--m_test mmap_data/weather_data_test.json
```

This fetches 42 years worth of data (from 1979 to 2020), 7 different weather variables and regrids it to 5.625, 2.8125 and 1.40625 deg

```sh
nohup bash scripts/fetch_42years_u_v_wind_spec_hum_rel_hum_t2m_tcc_tp.sh
```

This fetches 42 years worth of data (from 1979 to 2020), 2 weather variables and regrids it to 5.625, 2.8125 and 1.40625 deg

```sh
nohup bash scripts/fetch_42years_z500_850hPa.sh
```

To find out more information there is a help message on each of the scripts, to see the
message the script needs to be run with the `--help` flag

```sh
bash scripts/<script_name>.sh --help
```

---

## **Manual docker image compilation**

Parser container

```sh
docker build --build-arg secret_code=<secret_code> -t utils_parser docker/parser_docker
```

Torch Cuda training container

```sh
docker build -t ml_model docker/torch_docker
```

---

## **Manual execution with docker container**

To use any data parsing script with the parsing container

```
docker run --rm --init \
    --volume="$PWD:/app" \
    $container_name python3 <path> \
        <variables>
```

To use the taskgen.py in the Cuda training container

```sh
docker run --rm --init \
    --gpus=all \
    --ipc=host \
    --user="$(id -u):$(id -g)" \
    --volume="$PWD:/app" \
    ml_model python3 src/taskgen.py \
        <parameters>
```

---

## **Manual script execution**

This will download 6 different variables and 2 months (1416 samples) of ERA5 weather
maps from the CDS API. There needs to be a setup beforehand: [link](https://cds.climate.copernicus.eu/api-how-to)

```sh
python3 src/wb_utils/download.py single \
--variable 2m_temperature 10m_u_component_of_wind \
10m_v_component_of_wind total_cloud_cover total_precipitation \
toa_incident_solar_radiation \
--level_type single \
--output_dir data \
--custom_fn era5_data.nc \
--years 1979 \
--month 01 02
```

(Optional) - parses data to WeatherBench dataset resolution

```sh
python3 src/wb_utils/regrid.py \
--input_fns data/era5_data.nc \
--output_dir data \
--ddeg_out 2.8125 \
--custom_fn parsed_data.nc
```

Parses data to memmap format

```sh
python3 src/utils/parser.py \
--input_file data/parsed_data.nc \
--output_file mmap_data \
--output_fn weather_data \
--channels t2m u10 v10 tcc tp tisr \
--samples -1 \
--time_steps 1
```

Starts the training process of the model

```sh
python3 src/main.py --data_train mmap_data/weather_data_train.mmap \
--meta_train mmap_data/weather_data_train.json \
--data_test mmap_data/weather_data_test.mmap \
--meta_test mmap_data/weather_data_test.json
```
