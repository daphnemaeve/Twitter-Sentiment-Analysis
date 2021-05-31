#!/usr/bin/env python3

import csv
from csv import reader
from os import listdir,system
from functools import partial
from random import shuffle
import sys

percent = 0.80

parties = ["democrat", "republican", "green", "libertarian"]

def party_to_label(party):
    if party == "democrat":
        return 0
    elif party == "republican":
        return 1
    elif party == "green":
        return 2
    elif party == "libertarian":
        return 3
    else:
        print >> sys.stderr, ("unkown party: "+party)
        sys.exit(1)

def label_files(party,filename):
    return [(party_to_label(party),csv[3]) for csv in list(reader(open(party+'/'+filename, 'r')))[1:]]

data = [ list(map(partial(label_files,party), listdir(party)))
                        for party in parties ]

shuffle(data)
size = len(data)
training = data[0:int(size*percent)]
testing = data[int(size*percent):size]

system("rm {training.csv,testing.csv} &> /dev/null")
with open("training.csv", 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["label","text"])
    csvwriter.writerows(sum(sum(training,[]),[]))

with open("testing.csv", 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["label","text"])
    csvwriter.writerows(sum(sum(testing,[]),[]))
