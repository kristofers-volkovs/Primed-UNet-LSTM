#!/bin/bash

function show_usage (){
    printf "Usage: $0 [options [parameters]]\n"
    printf "\n"
    printf "Options:\n"
    printf " -n|--name, Set the created container name, default: utils_parser\n"
    printf " -p|--path, Path to Dockerfile\n"
    printf " -c|--code, secret API code from the CDS webpage: https://cds.climate.copernicus.eu/api-how-to (Optional)\n"
    printf " --ignore_code, ignores if secret code was not passed, default = true (optional)\n"
    printf " -r|--recompile, Recompiles the docker image even if it has already been compiled before\n"
    printf " --input_dirss, Input file directories\n"
    printf " --output_dir, Output directory\n"
    printf " --channels, All data file variables that need to be parsed to the new Memmap file\n"
    printf " --samples, Amount of samples that are gonna be copied (default = -1, selects all samples from data file)\n"
    printf " --time_steps, Amount of time steps (default = 1, t_0 and t_1 will be encoded)\n"
    printf " --delta, Time between each time step (default = 1)\n"
    printf " --shuffle, Shuffle data along the sample axis (default = True)\n"
    printf " --normalize, Normalize data or not (default = True)\n"
    printf " --output_fn, If not None, use custom file name, default: parsed_data.nc\n"
    printf " --rnn, Creates mmap data shapes used for rnn model, default: False\n"
    printf " -h|--help, Displays the help message\n"

    return 0
}

container_name='utils_parser'
recompile=False
output_dir='mmap_data'
output_fn='weather_data'
ignore_code=Frue

while [ ! -z "$1" ]; do
    case "$1" in
        --name|-n)
            shift
            container_name="$1"
            ;;
        --path|-p)
            shift
            dockerfile_path="$1"
            ;;
        --code|-c)
            shift
            secret_code="$1"
            ;;
        --ignore_code|-c)
            shift
            ignore_code="$1"
            ;;
        --recompile|-r)
            shift
            recompile="$1"
            ;;
        --output_fn)
            shift
            output_fn="$1"
            ;;
        --input_dirs)
            input_dirs=()
            idx=0
            while [[ ! "$2" =~ ^-{1,2}\w* ]] && [ ! -z "$2" ]; do
                shift
                input_dirs[idx]="$1"
                idx=$((idx+1))
            done
            ;;
        --output_dir)
            shift
            output_dir="$1"
            ;;
        --data_type)
            shift
            data_type="$1"
            ;;
        --channels)
            channels=()
            idx=0
            while [[ ! "$2" =~ ^-{1,2}\w* ]] && [ ! -z "$2" ]; do
                shift
                channels[idx]="$1"
                idx=$((idx+1))
            done
            ;;
        --samples)
            shift
            samples="$1"
            ;;
        --time_steps)
            shift
            time_steps="$1"
            ;;
        --delta)
            shift
            delta="$1"
            ;;
        --shuffle)
            shift
            shuffle="$1"
            ;;
        --normalize)
            shift
            normalize="$1"
            ;;
        --rnn)
            shift
            rnn="$1"
            ;;
        --help|-h)
            show_usage
            exit
            ;;
        *)
            printf "Wrong argument/s: '$1'\n\n"
            show_usage
            exit
            ;;
    esac
shift
done

if [ -z ${dockerfile_path+x} ] || [ -z ${input_dirs+x} ] || [ -z ${output_dir+x} ]; then
    printf "Some argument/s was/were not set properly\n"
    printf "Obligatory arguments:\n"
    printf "dockerfile_path: $dockerfile_path\n"
    printf "input_dirs: $input_dirs\n"
    printf "output_dir: $output_dir\n"
    exit
fi

printf "\nContainer name is set to '$container_name'\n"
printf "Output dir is set to: '$output_dir'\n"
printf "Resulting train data file name: '${output_fn}_train.mmap'\n"
printf "Resulting test data file name: '${output_fn}_test.mmap'\n"

result=$( docker images -q $container_name )

if [ -n "$result" ] && ! [ $recompile = true ]; then
    printf '\n---------- Image already exists ----------\n\n'
else
    printf '\n---------- Compiling docker image from Dockerfile ----------\n\n'

    if [ -z ${secret_code+x} ] && [ $ignore_code = false ]; then
        printf "The secret code was not provided\n"
        exit
    fi

    # Builds docker image
    docker build --build-arg secret_code=$secret_code -t $container_name $dockerfile_path
fi

idx=0
if [ ! -z ${samples+x} ]; then
    optional_parameters[idx]="--samples ${samples}"
    idx=$((idx+1))
fi
if [ ! -z ${time_steps+x} ]; then
    optional_parameters[idx]="--time_steps ${time_steps}"
    idx=$((idx+1))
fi
if [ ! -z ${delta+x} ]; then
    optional_parameters[idx]="--delta ${delta}"
    idx=$((idx+1))
fi
if [ ! -z ${shuffle+x} ]; then
    optional_parameters[idx]="--shuffle ${shuffle}"
    idx=$((idx+1))
fi
if [ ! -z ${normalize+x} ]; then
    optional_parameters[idx]="--normalize ${normalize}"
    idx=$((idx+1))
fi
if [ ! -z ${rnn+x} ]; then
    optional_parameters[idx]="--rnn ${rnn}"
    idx=$((idx+1))
fi
if [ ! -z ${data_type+x} ]; then
    optional_parameters[idx]="--data_type ${data_type}"
    idx=$((idx+1))
fi

idx=0
input_files=()
for dir in "${input_dirs[@]}"
do
    for entry in "$dir"/*
    do
        input_files[idx]=$entry
        idx=$((idx+1))
    done
done

# Runs the regrid script through the docker container
docker run --rm --init \
    --volume="$PWD:/app" \
    $container_name python3 src/utils/parser.py \
        --input_file ${input_files[@]} \
        --output_file $output_dir \
        --output_fn $output_fn \
        --channels ${channels[@]} \
        ${optional_parameters[*]}
