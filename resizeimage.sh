#!/bin/bash

# get the current directory
#cd "$( dirname "${BASH_SOURCE[0]}" )"
DIR="data/Caltech-101"
classcnt=0
if [ -f "temp.txt" ]; then
	rm temp.txt
else
	touch temp.txt
fi

for class in `ls $DIR`; do
	if [ -d ${DIR}/${class} ]; then
		for image in `ls ${DIR}/${class}`; do
			if [[ $image == *"jpg" ]]; then
				# resize image 
				#convert -resize 227x227\! ${class}/${image} ${class}/${image}
				echo "${DIR}/${class}/${image} $classcnt" >> temp.txt
			fi
		done
		classcnt=$((classcnt+1))
	fi	
done

