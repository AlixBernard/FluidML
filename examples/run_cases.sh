#!/usr/bin/env bash
# @Author: AlixBernard
# @Email: alix.bernard9@gmail.com
# @Date: 2022-06-06 22:28:00
# @Last modified by: AlixBernard
# @Last modified time: 2022-06-07 19:08:17

cases=(
	'PH-Re10595_CD-Re12600'
	'PH-Re10595_SD-Re2900'
	'PH-Re10595_CD-Re12600_SD-Re2900'
	'SD-Re2900_SD-Re3500'
)

for case in ${cases[*]}; do
	if [ $# == 0 ]; then
		echo -e "########################################\nTrain case ${case}\n"
		time python "$case/train.py"
		echo -e "########################################\nPlot case ${case}\n"
		time python "$case/plot_b.py"
	elif [ $1 == '--train' ]; then
		echo -e "########################################\nTrain case ${case}\n"
		time python "$case/train.py"
	elif [ $1 == '--plot' ]; then
		echo -e "########################################\nPlot case ${case}\n"
		time python "$case/plot_b.py"
	else
		echo "The argument ${1} is not recognized"
		echo -e 'Possible arguments are:\n\t--train\n\t--plot'
		exit
	fi
done
