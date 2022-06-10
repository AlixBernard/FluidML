#!/usr/bin/env bash
# @Author: AlixBernard
# @Email: alix.bernard9@gmail.com
# @Date: 2022-06-06 21:17:36
# @Last modified by: AlixBernard
# @Last modified time: 2022-06-07 09:38:27

template_folder='template_case'
new_folders=(
	'PH-Re10595_CD-Re12600'
	'PH-Re10595_SD-Re2900'
	'PH-Re10595_CD-Re12600_SD-Re2900'
	'SD-Re2900_SD-Re3500'
)

echo "Copying '${template_folder}' as:"
for folder in ${new_folders[*]}; do
	rm -r "${folder}"
	cp -r "${template_folder}" "${folder}"
	echo "    '${folder}'"
done
