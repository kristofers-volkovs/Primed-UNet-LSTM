#!/bin/bash

function show_usage (){
    printf "Usage: $0 [options [parameters]]\n"
    printf "\n"
    printf "Options:\n"
    printf " -n|--name, Set the created container name, default: utils_parser\n"
    printf " -p|--path, Path to Dockerfile\n"
    printf " -c|--code, secret API code from the CDS webpage: https://cds.climate.copernicus.eu/api-how-to\n"
    printf " --ignore_code, ignores if secret code was not passed (optional)\n"
    printf " -r|--recompile, Recompiles the docker image even if it has already been compiled before\n"
    printf " --mode, 'single' or 'separate'. If 'several', loops over years, default: single\n"
    printf " --variable, Name of variable(s) in archive\n"
    printf " --level_type, 'single' or 'pressure'\n"
    printf " --pressure_level, Pressure levels to download. None for 'single' output type (optional)\n"
    printf " --output_dir, Directory where file is stored\n"
    printf " --years, Years to download, each year is saved separately if mode is 'separate\n"
    printf " --month, Month(s) to download, default: selects all available months\n"
    printf " --day, Day(s) to download, default: selects all available days\n"
    printf " --time, Time(s) to download, default: selects all available time\n"
    printf " --custom_fn, If not None, use custom file name, default: era5_data_<year>.nc\n"
    printf " -h|--help, Displays the help message\n"

    return 0
}

container_name='utils_parser'
recompile=false
mode=single
ignore_code=true
custom_fn='era5_data'

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
        --mode)
            shift
            mode="$1"
            ;;
        --variable)
            variable=()
            idx=0
            while [[ ! "$2" =~ ^-{1,2}\w* ]] && [ ! -z "$2" ]; do
                shift
                variable[idx]="$1"
                idx=$((idx+1))
            done
            ;;
        --level_type)
            shift
            level_type="$1"
            ;;
        --pressure_level)
            pressure_level=()
            idx=0
            while [[ ! "$2" =~ ^-{1,2}\w* ]] && [ ! -z "$2" ]; do
                shift
                pressure_level[idx]="$1"
                idx=$((idx+1))
            done
            ;;
        --output_dir)
            shift
            output_dir="$1"
            ;;
        --years)
            years=()
            idx=0
            while [[ ! "$2" =~ ^-{1,2}\w* ]] && [ ! -z "$2" ]; do
                shift
                years[idx]="$1"
                idx=$((idx+1))
            done
            ;;
        --month)
            month=()
            idx=0
            while [[ ! "$2" =~ ^-{1,2}\w* ]] && [ ! -z "$2" ]; do
                shift
                month[idx]="$1"
                idx=$((idx+1))
            done
            ;;
        --day)
            day=()
            idx=0
            while [[ ! "$2" =~ ^-{1,2}\w* ]] && [ ! -z "$2" ]; do
                shift
                day[idx]="$1"
                idx=$((idx+1))
            done
            ;;
        --time)
            time=()
            idx=0
            while [[ ! "$2" =~ ^-{1,2}\w* ]] && [ ! -z "$2" ]; do
                shift
                time[idx]="$1"
                idx=$((idx+1))
            done
            ;;
        --custom_fn)
            shift
            custom_fn="$1"
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

if [ -z ${dockerfile_path+x} ] || [ -z ${output_dir+x} ] || [ ${#variable[@]} -lt 0 ] || [ -z ${level_type+x} ] || [ ${#years[@]} -lt 0 ]; then
    printf "Some argument/s was/were not set properly\n"
    printf "Obligatory arguments:"
    printf "dockerfile_path: $dockerfile_path"
    printf "output_dir: $output_dir\n"
    printf "variable: ${variable[@]}"
    printf "level_type: $level_type"
    printf "years: ${years[@]}"
    exit
fi

printf "\nContainer name is set to '$container_name'\n"
printf "Output dir is set to: '$output_dir'\n"
printf "Resulting file name: '$custom_fn'\n"

result=$( docker images -q $container_name )

if [ -n "$result" ] && ! [ $recompile = true ]; then
    printf '\n---------- Image already exists ----------\n'
else
    printf '\n---------- Compiling docker image from Dockerfile ----------\n'

    if [ -z ${secret_code+x} ] && [ $ignore_code = false ]; then
        printf "The secret code was not provided\n"
        exit
    fi

    # Builds docker image
    docker build --build-arg secret_code=$secret_code -t $container_name $dockerfile_path
fi

printf '\n---------- Starting the data fetching ----------\n\n'

idx=0
if [ ${#pressure_level[@]} -gt 0 ]; then
    optional_parameters[idx]="--pressure_level ${pressure_level[*]}"
    idx=$((idx+1))
fi
if [ ${#month[@]} -gt 0 ]; then
    optional_parameters[idx]="--month ${month[*]}"
    idx=$((idx+1))
fi
if [ ${#day[@]} -gt 0 ]; then
    optional_parameters[idx]="--day ${day[*]}"
    idx=$((idx+1))
fi
if [ ${#time[@]} -gt 0 ]; then
    optional_parameters[idx]="--time ${time[*]}"
    idx=$((idx+1))
fi

if [ ${#years[@]} -gt 0 ]; then
    # Runs the download script through the docker container
    # Fetches each year sepperately
    for year in ${years[@]};
    do
        file_name="${custom_fn}_${year}.nc"

        docker run --rm --init \
        --volume="$PWD:/app" \
        $container_name python3 src/wb_utils/download.py $mode \
            --variable ${variable[*]} \
            --level_type $level_type \
            --output_dir $output_dir \
            --custom_fn $file_name \
            --years $year \
            ${optional_parameters[*]}
    done
fi
