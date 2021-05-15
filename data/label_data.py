#!/usr/bin/env python3

import csv
from csv import reader
from os import listdir
from functools import partial

parties = ["democrat", "republican", "green", "libertarian"]

def label_files(party,filename):
    return [(party,csv[3]) for csv in list(reader(open(party+'/'+filename, 'r')))[1:]]

party_files = [ list(map(partial(label_files,party), listdir(party)))
                for party in parties ]

with open("labeled_data.csv", 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["party","tweet"])
    csvwriter.writerows(sum(sum(party_files,[]),[]))
