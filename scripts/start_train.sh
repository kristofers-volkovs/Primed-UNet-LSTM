#!/bin/bash

function show_usage (){
    printf "Usage: $0 [options [parameters]]\n"
    printf "\n"
    printf "Options:\n"
    printf " -n|--name, Set the created container name, default: utils_wb\n"
    printf " -p|--path, Path to Dockerfile\n"
    printf " -r|--recompile, Recompiles the docker image even if it has already been compiled before\n"
    printf " --d_train, training data in memmap format\n"
    printf " --m_train, training data metadata, json\n"
    printf " --d_test, testing data in memmap format\n"
    printf " --m_test, testing data metadata, json\n"
    printf " -h|--help, Displays the help message\n"

    return 0
}

container_name='ml_model'
recompile=false

# TODO add hyper params
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
        --d_train)
            shift
            data_train="$1"
            ;;
        --m_train)
            shift
            meta_train="$1"
            ;;
        --d_test)
            shift
            data_test="$1"
            ;;
        --m_test)
            shift
            meta_test="$1"
            ;;
        --recompile|-r)
            shift
            recompile="$1"
            ;;
        --help|-h)
            show_usage
            ;;
        *)
            printf "Wrong argument/s: '$1'\n\n"
            show_usage
            exit
            ;;
    esac
shift
done

if [ -z ${dockerfile_path+x} ] || [ -z ${data_train+x} ] || [ -z ${meta_train+x} ] || [ -z ${data_test+x} ] || [ -z ${meta_test+x} ]; then
    printf "Some argument/s was/were not set properly\n"
    printf "Obligatory arguments:\n"
    printf "dockerfile_path: $dockerfile_path\n"
    printf "d_train: $data_train\n"
    printf "m_train: $meta_train\n"
    printf "d_test: $data_test\n"
    printf "m_test: $meta_test\n"
    exit
fi

printf "\nContainer name is set to '$container_name'"

result=$( docker images -q $container_name )

if [ -n "$result" ] && ! [ $recompile = true ]; then
    printf "\n---------- Image already exists ----------\n"
else
    printf '\n---------- Compiling docker image from Dockerfile ----------\n'

    # Builds docker image
    docker build -t $container_name $dockerfile_path
fi

printf '\n---------- Starting the training process ----------\n\n'

# Runs the training script through the docker container
docker run --rm --init \
    --gpus=all \
    --ipc=host \
    --user="$(id -u):$(id -g)"\
    --volume="$PWD:/app"\
    $container_name python3 src/main.py --data_train $data_train \
        --meta_train $meta_train \
        --data_test $data_test \
        --meta_test $meta_test
