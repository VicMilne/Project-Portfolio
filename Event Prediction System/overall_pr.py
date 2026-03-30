# compute overall player rankings as sum of every individual contest's rankings
# weighted by contest frequency

import numpy as np
import os.path

from C1.Model_C1 import C1_class
from C2.Model_C2 import C2_class
from C3.Model_C3 import C3_class
from C4.Model_C4 import C4_class
from C5.Model_C5 import C5_class
from C6.Model_C6 import C6_class
from C7.Model_C7 import C7_class
from C8.Model_C8 import C8_class

# dictionary pairing contest name with the class instances
contests = {"C1": [C1_class()], "C2": [C2_class()], "C3": [C3_class()], "C4": [C4_class()], "C5": [C5_class()],
         "C6": [C6_class()], "C7": [C7_class()], "C8": [C8_class()]}

f = open("current_contest_freqs.txt", 'r')
txt = f.read()
f.close()

freqs = txt.split("\n")[0]

# use ordered property of python dictionaries to associate each contest with a weight
freqs = freqs.split("\t")
for i, contest in enumerate(contests):
    if(i != len(freqs)):
        contests[contest].append(float(freqs[i]))


pr_load = {}
for contest in contests:
    contest_pr = contests[contest][0].pr_results()

    string = ""
    for player in contest_pr: string += (player[0] + "\t" + str(player[1]) + "\n")

    for pla, score in contest_pr:
        if pla not in pr_load:
            pr_load[pla] = [0,0]
        pr_load[pla][0] += score * contests[contest][1] / 8
        pr_load[pla][1] += contests[contest][1] / 8

pr = []
for pla in pr_load:
    pr.append([pla, round(pr_load[pla][0] * 8 / pr_load[pla][1])]) 

pr = (sorted(pr,key=lambda x: (x[1])))
pr.reverse()

string = ""
for player in pr:
    string = string + (player[0] + "\t" + str(player[1]) + "\n")

check = 0
    


