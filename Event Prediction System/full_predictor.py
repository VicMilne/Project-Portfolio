import numpy as np

from C1.Model_C1 import C1_class
from C2.Model_C2 import C2_class
from C3.Model_C3 import C3_class
from C4.Model_C4 import C4_class
from C5.Model_C5 import C5_class
from C6.Model_C6 import C6_class
from C7.Model_C7 import C7_class
from C8.Model_C8 import C8_class

# define contests and teams for event
new_event_contests = ["C1", "C3", "C4", "C6", "C7", "C8"]

new_roster = [["P020", "P029", "P025", "P051"], ["P001", "P049", "P019", "P203"],
              ["P068", "P002", "P038", "P073"], ["P046", "P011", "P064", "P027"],
              ["P028", "P227", "P210", "P026"], ["P036", "P053", "P005", "P225"],
              ["P042", "P144", "P016", "P015"], ["P034", "P231", "P023", "P213"],
              ["P022", "P009", "P060", "P226"], ["P000", "P062", "P009", "P202"]
]

# dictionary pairing contest name with the class instances
contests = {"C1": [C1_class()], "C2": [C2_class()], "C3": [C3_class()], "C4": [C4_class()], "C5": [C5_class()],
         "C6": [C6_class()], "C7": [C7_class()], "C8": [C8_class()]
        }

f = open("current_contest_freqs.txt", 'r')
txt = f.read()
f.close()

adjs = txt.split("\n")[1]

# use ordered property of python dictionaries to associate each contest with a weight
adjs = adjs.split("\t")
for i, contest in enumerate(contests):
    if(i != len(adjs)):
        contests[contest].append(float(adjs[i]))

# initialize score arrays for each team
un_adj = [0,0,0,0,0,0,0,0,0,0]
overall_adj = [0,0,0,0,0,0,0,0,0,0]

# initialize indiv list for individual scores
indiv = []
for team in new_roster:
    for pla in team:
        indiv.append([pla, 0])

tot_adj = 0
for contest in new_event_contests:

    results = contests[contest][0].cur_sim(new_roster)

    # results are returned as 40 individual results (team by team)
    for m in range(40):
        indiv[m][1] += results[m]
    
    # compute team results by summing in groups of 4
    ten_results = []
    for j in range(10):
        ten_results.append(sum(results[j*4:(j+1)*4]))

    # create formatted strings with the results for easy extraction
    str1 = ""
    str2 = ""
    for j in range(len(results)):
        str1 += str(results[j]) + "\n"
    for j in range(len(ten_results)):
        str2 += str(ten_results[j]) + "\n"
    
    # add the team results to un_adj and overall_adj (multiply by contest weight)
    un_adj = np.add(un_adj, ten_results)
    adj = contests[contest][1]
    ten_results = [x * adj for x in ten_results]
    tot_adj += adj
    overall_adj = np.add(overall_adj, ten_results)

# divide by number of contests, always adjust as if 6 contests are present
un_adj = [round(x*6/len(new_event_contests)) for x in un_adj]

# additional adjustment required for overall_adj, divide by sum of weights
overall_adj = [round(x*6/tot_adj) for x in overall_adj]

# create formatted strings with the results for easy extraction
str1 = "\n".join([str(x) for x in un_adj])
str2 = "\n".join([str(x) for x in overall_adj])
str3 = "\n".join([str(x[1]) for x in indiv])

# sort indiv scoring for inspection
indiv = (sorted(indiv,key=lambda x: (x[1])))
indiv.reverse()
check = 0

check = 1