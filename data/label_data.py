#!/usr/bin/env python3

import csv
from csv import reader
from os import listdir,system
from functools import partial
from random import shuffle
import sys
import re

# ratio of training/testing
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

# lines that are empty are removed
def sanitize_line(t):
    party = t[0]
    text = t[1]
    text = text.replace('\n',' ')
    text = text.replace('""','"')
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"RT .*: .*",'',text)
    return [party,str(text)]

data = [ list(map(partial(label_files,party), listdir(party))) for party in parties ]
data = sum(sum(data,[]),[])
shuffle(data)
data = map(sanitize_line,data)
data = list(filter(lambda x: x[1] != ' ', data))
size = len(data)
training = data[0:int(size*percent)]
testing = data[int(size*percent):size]

system("rm {training.csv,testing.csv} &> /dev/null")
with open("training.csv", 'w') as csvfile:
    csvwriter = csv.writer(csvfile,quoting=csv.QUOTE_NONNUMERIC)
    csvwriter.writerow(["label","text"])
    csvwriter.writerows(training)

with open("testing.csv", 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["label","text"])
    csvwriter.writerows(testing)
