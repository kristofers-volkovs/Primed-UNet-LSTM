#!/bin/bash

function show_usage (){
    printf "Usage: $0 [options [parameters]]\n"
    printf "Used for fetching 40 years worth of data (from 1979 to 2020) and regrids it to 5.625, 2.8125 and 1.40625 deg"
    printf "\n"
    printf "Options:\n"
    printf " -h|--help, Displays the help message\n"

    return 0
}

container_name='utils_parser'

if [ "$1" = --help ] || [ "$1" = -h ]; then
    show_usage
    exit
fi

printf '\n---------- Starting the data fetching ----------\n\n'

bash scripts/fetch.sh -p docker/parser_docker \
    --mode single \
    --variable geopotential \
    --level_type pressure \
    --pressure_level 500 \
    --output_dir data/1979_2020_z500_t850 \
    --custom_fn era5_z500 \
    --years 1979 1980 1981 1982 1983 1984 1985 1986 1987 1988 1989 1990 1991 1992 1993 1994 1995 1996 1997 1998 1999 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020

bash scripts/fetch.sh -p docker/parser_docker \
    --mode single \
    --variable temperature \
    --level_type pressure \
    --pressure_level 850 \
    --output_dir data/1979_2020_z500_t850 \
    --custom_fn era5_t850 \
    --years 1979 1980 1981 1982 1983 1984 1985 1986 1987 1988 1989 1990 1991 1992 1993 1994 1995 1996 1997 1998 1999 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020

printf '\n---------- Data was successfuly fetched ----------\n'
printf '\n---------- Starting data regriding to 5.625 deg ----------\n\n'

bash scripts/parse_wb.sh -p docker/parser_docker \
    --input_dir data/1979_2020_z500_t850 \
    --output_dir data/regrid_1979_2020_z500_t850_5.625deg \
    --ddeg_out 5.625

printf '\n---------- Data was successfuly fetched ----------\n'
printf '\n---------- Starting data regriding to 2.8125 deg ----------\n\n'

bash scripts/parse_wb.sh -p docker/parser_docker \
    --input_dir data/1979_2020_z500_t850 \
    --output_dir data/regrid_1979_2020_z500_t850_2.8125deg \
    --ddeg_out 2.8125

printf '\n---------- Data was successfuly fetched ----------\n'
printf '\n---------- Starting data regriding to 1.40625 deg ----------\n\n'

bash scripts/parse_wb.sh -p docker/parser_docker \
    --input_dir data/1979_2020_z500_t850 \
    --output_dir data/regrid_1979_2020_z500_t850_1.40625deg \
    --ddeg_out 1.40625

printf '\n---------- Data was successfuly parsed ----------\n\n'
