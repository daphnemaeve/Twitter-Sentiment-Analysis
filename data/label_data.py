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

training_size = 10000
testing_size = 2500

large_training = 100000
large_testing = 25000

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
    text = text.replace('\n',' ') # replace new lines with space
    text = re.sub(r"http\S+", "", text) # remove links
    text = re.sub(r"RT .*: .*",'',text) # remove retweets
    text = re.sub(r"@[\w]+",'',text) # remove @user
    text = text.replace('""','"') # remove double quotes
    text = re.sub(r"[\s]+",' ',text) # remove duplicate spaces
    return [party,text]

data = [ list(map(partial(label_files,party), listdir(party))) for party in parties ]
data = sum(sum(data,[]),[])
shuffle(data)
data = map(sanitize_line,data)
data = list(filter(lambda x: x[1] != ' ', data))

size = len(data)
training = data[0:int(size*percent)]
testing = data[int(size*percent):size]

large_training = training[0:large_training]
large_testing = testing[0:large_testing]

training = training[0:training_size]
testing = testing[0:testing_size]

system("rm {training.csv,testing.csv} &> /dev/null")
with open("training.csv", 'w') as csvfile:
    csvwriter = csv.writer(csvfile,quoting=csv.QUOTE_NONNUMERIC)
    csvwriter.writerow(["label","text"])
    csvwriter.writerows(training)

with open("testing.csv", 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["label","text"])
    csvwriter.writerows(testing)

system("rm {large_training.csv,large_testing.csv} &> /dev/null")
with open("large_training.csv", 'w') as csvfile:
    csvwriter = csv.writer(csvfile,quoting=csv.QUOTE_NONNUMERIC)
    csvwriter.writerow(["label","text"])
    csvwriter.writerows(large_training)

with open("large_testing.csv", 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["label","text"])
    csvwriter.writerows(large_testing)
