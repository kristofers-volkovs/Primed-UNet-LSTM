#!/bin/bash

function show_usage (){
    printf "Usage: $0 [options [parameters]]\n"
    printf "\n"
    printf "Options:\n"
    printf " -n|--name, Set the created container name, default: utils_parser\n"
    printf " -p|--path, Path to Dockerfile\n"
    printf " -c|--code, secret API code from the CDS webpage: https://cds.climate.copernicus.eu/api-how-to (optional)\n"
    printf " --ignore_code, ignores if secret code was not passed (optional)\n"
    printf " -r|--recompile, Recompiles the docker image even if it has already been compiled before\n"
    printf " --input_dir, input directory, all files are parsed in the directory\n"
    printf " --output_dir, Output directory\n"
    printf " --ddeg_out, Output resolution, WeatherBench options: 1.40625, 2.8125 or 5.625\n"
    printf " --file_ending, File ending, default = nc\n"
    printf " --is_grib, Input is .grib file. 0 (default) or 1\n"
    printf " --custom_fn, If not None, use custom file name\n"
    printf " -h|--help, Displays the help message\n"

    return 0
}

container_name='utils_parser'
recompile=false
ignore_code=true
reuse_weights=0

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
        --custom_fn)
            shift
            custom_fn="$1"
            ;;
        --input_dir)
                shift
            input_dir="$1"
            ;;
        --output_dir)
            shift
            output_dir="$1"
            ;;
        --ddeg_out)
            shift
            ddeg_out="$1"
            ;;
        --file_ending)
            shift
            file_ending="$1"
            ;;
        --is_grib)
            shift
            is_grib="$1"
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

if [ -z ${dockerfile_path+x} ] || [ -z ${input_dir+x} ] || [ -z ${output_dir+x} ] || [ -z ${ddeg_out+x} ]; then
    printf "Some argument/s was/were not set properly\n"
    printf "Obligatory arguments:\n"
    printf "dockerfile_path: $dockerfile_path\n"
    printf "input_dir: $input_dir\n"
    printf "output_dir: $output_dir\n"
    printf "ddeg_out: $ddeg_out\n"
    exit
fi

printf "\nContainer name is set to '$container_name'\n"
printf "Output dir is set to: '$output_dir'\n"

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
if [ ! -z ${custom_fn+x} ]; then
    optional_parameters[idx]="--custom_fn ${custom_fn}"
    idx=$((idx+1))
fi
if [ ! -z ${is_grib+x} ]; then
    optional_parameters[idx]="--is_grib ${is_grib}"
    idx=$((idx+1))
fi
if [ ! -z ${file_ending+x} ]; then
    optional_parameters[idx]="--file_ending ${file_ending}"
    idx=$((idx+1))
fi

idx=0
for entry in "$input_dir"/*
do
    # Runs the regrid script through the docker container
    docker run --rm --init \
        --volume="$PWD:/app" \
        $container_name python3 src/wb_utils/regrid.py \
            --input_fns $entry \
            --output_dir $output_dir \
            --ddeg_out $ddeg_out \
            --reuse_weights $reuse_weights \
            --custom_fn "${idx}_${ddeg_out}deg.nc" \
            ${optional_parameters[*]}
    idx=$((idx+1))
done

printf "\nData was succesfully parsed \n"
