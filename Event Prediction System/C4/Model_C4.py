import numpy as np
import scipy.optimize as sci
import os.path

##############################################################################################

# Consistently Updated Globals and Classes

default_decay = 0.95
default_d = []

e_num = 28

# numbered events where this contest appeared
event_select = [3,4,5,6,7,8,9,11,12,14,15,16,17,19,20,22,23,24,25,27,29,31,32,34,37,39,41,42]

# team by team results for all events
results = [1228, 884, 716, 1504, 732, 764, 1348, 292, 1316, 1196, 1244, 876, 1380, 1552, 192, 1564, 620, 724, 792, 1120,
    508, 740, 1668, 1036, 1112, 1264, 1484, 676, 300, 1280, 1392, 300, 1640, 724, 1052, 528, 876, 1216, 888, 1396, 
    1884, 1068, 848, 592, 788, 1548, 1456, 224, 848, 748, 520, 932, 1516, 1388, 1072, 468, 1660, 1144, 664, 812, 
    1128, 768, 1212, 1168, 884, 1124, 832, 1416, 824, 716, 1108, 1124, 1380, 1124, 132, 1436, 1384, 396, 968, 800, 
    1484, 1024, 956, 936, 560, 1128, 1108, 1468, 764, 628, 1060, 852, 768, 888, 952, 1128, 1264, 1212, 1096, 848, 
    1108, 748, 936, 660, 1076, 1196, 1128, 868, 1024, 1316, 1368, 1160, 988, 972, 1280, 1248, 840, 360, 1108, 392, 
    880, 784, 960, 876, 868, 1360, 1460, 1024, 760, 1096, 852, 1012, 1340, 1192, 760, 1144, 588, 704, 1644, 816, 
    960, 1308, 1160, 880, 1492, 596, 1040, 636, 992, 944, 716, 504, 1564, 1212, 876, 668, 1476, 1092, 832, 1128, 
    1092, 960, 1360, 884, 1012, 1144, 1424, 604, 1092, 488, 1372, 1004, 644, 900, 856, 996, 1312, 1392, 1264, 240, 
    1124, 1128, 1648, 628, 596, 912, 572, 1340, 1208, 908, 840, 1392, 1072, 868, 1112, 1356, 828, 184, 1328, 976, 
    892, 1412, 1176, 716, 900, 840, 1176, 792, 640, 1528, 580, 596, 1512, 1308, 976, 420, 1164, 976, 1176, 1072, 
    1380, 1272, 916, 956, 1028, 1324, 776, 808, 808, 796, 528, 1272, 840, 1196, 892, 1492, 988, 680, 408, 1296, 
    1208, 1296, 1592, 840, 256, 1284, 580, 1272, 936, 592, 624, 1612, 840, 840, 1212, 764, 620, 1024, 1376, 1096, 
    1544, 1224, 1232, 1108, 476, 1056, 808, 544, 632, 1096, 1696, 1358, 1312, 1145, 988, 977, 738, 781, 693, 418
    ]

# baseline predictions to compare against
pr_preds = [929,1146,1327,1117,640,893,1384,1092,745,768,1047,495,1101,1120,710,1408,1039,949,1011,1161,
    1149,787,1224,801,1108,1265,954,1158,1027,569,1137,1188,1101,1036,1127,1072,988,415,1121,856,
    1081,737,1370,922,1310,1085,1149,607,409,1302,1180,1121,1185,443,693,1119,1371,1189,510,1158,
    710,1452,1127,730,1139,662,616,1322,1006,1227,884,886,1604,869,894,646,1019,1201,655,1331,
    923,684,781,1151,547,1355,857,1424,946,1321,1106,882,905,643,922,1012,960,1717,1077,655,
    937,716,1073,883,731,1379,1525,736,1214,688,1070,1026,1096,1262,867,1260,1124,200,998,986,
    867,1162,1182,1034,800,1209,1165,1147,810,1119,1072,1080,1415,1099,200,944,737,1068,1406,875,
    631,1186,1445,1426,917,1298,927,380,736,950,1249,1376,659,1028,845,434,784,1045,892,1588,
    794,948,1449,1153,200,1006,917,1169,1085,1086,945,1246,1018,574,887,729,578,1500,806,1525,
    990,990,990,990,990,990,990,990,990,990,990,990,990,990,990,990,990,990,990,990]

# denotes outlier events
outlier = [1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,1,0,1,1,1,0,0,1,1,0,0]


player_subs = {"P135": "P035", "P004": "P062", "P047": "P057", "P099": "P099", "P097": "P097",
"P046": "P001", "P016": "P035", "P026": "P099", "P134": "P032", "P130": "P024", "P056": "P010", "P006": "P017",
"P034": "P045", "P126": "P035", "P008": "P096", "P041": "P029", "P021": "P043", "P136": "P127", "P024": "P035",
"P052": "P024", "P028": "P074", "P005": "P016", "P002": "P008", "P030": "P026", "P009": "P008",
"P063": "P035", "P133": "P008", "P020": "P046", "P059": "P061", "P060": "P035", "P078": "P053", "P050": "P011",
"P124": "P044", "P064": "P024", "P000": "P049", "P125": "P006", "P017": "P024", "P137": "P060", "P039": "P144",
"P073": "P056", "P065": "P001", "P015": "P056", "P025": "P028", "P040": "P032", "P027": "P010",
"P075": "P056", "P061": "P059", "P132": "P017", "P115": "P118", "P096": "P127", "P086": "P033", "P120": "P017",
"P121": "P026", "P037": "P043", "P106": "P144", "P071": "P026", "P150": "P097", "P170": "P099",
"P171": "P097", "P172": "P099", "P167": "P096", "P168": "P006", "P117": "P057", "P107": "P035",
"P152": "P097", "P148": "P040", "P143": "P035", "P141": "P056", "P145": "P034", "P071": "P026",
"P153": "P027", "P154": "P097", "P147": "P032", "P155": "P096", "P151": "P097", "P149": "P040",
"P146": "P006", "P105": "P096", "P129": "P033", "P007": "P097", "P139": "P024", "P079": "P051", 
"P003": "P097", "P156": "P096", "P138": "P026", "P077": "P030", "P140": "P097", "P142": "P042",
"P069": "P154", "P197": "P076", "P198": "P063", "P199": "P006", "P118": "P030", "P098": "P047",
"P068": "P034", "P076": "P075", "P109": "P097", "P202": "P056", "P215": "P097", "P216": "P069",
"P217": "P076", "P218": "P097", "P219": "P000", "P220": "P002", "P221": "P016", "P213": "P026",
"P210": "P063", "P222": "P057", "P204": "P038", "P228": "P039", "P229": "P077", "P223": "P032",
"P230": "P016", "P224": "P012", "P225": "P073", "P226": "P015", "P227": "P144", "P214": "P036",
"P231": "P032", "P203": "P141", "P232": "P202", "P186": "P016", "P233": "P057", "P072": "P060",
"P200": "P002", "P208": "P027"}

new_players = {"P121", "P120", "P075", "P135", "P117", "P107", "P168", "P153", "P071", "P129", "P150", "P138", 
               "P076", "P197", "P218", "P215", "P139", "P203"}
non_players = {"P097", "P099", "P096", "P130", "P170", "P167", "P171", "P172", "P154",
               "P152", "P155", "P151", "P105", "P007", "P003", "P156",
               "P140", "P069", "P229"}

class C4_class:
    def __init__(self):
        self.pdict, self.popt = fill_pdict()
        self.e_dict = {"EVE_12": 7, "EVE_10": 6, "EVE_9": 5, "EVE_8": 4, "EVE_7": 3, "EVE_6": 2, "EVE_5": 1, "EVE_4": 0,
                       "EVE_13": 8, "EVE_14": 9, "EVE_15": 9, "EVE_16": 10, "EVE_18": 11, "EVE_19": 12, 
                       "EVE_21": 13, "EVE_22": 14, "EVE_24": 15, "EVE_25": 16, "EVE_26": 17, "EVE_27": 18,
                       "EVE_29": 19, "EVE_31": 20, "EVE_33": 21, "EVE_34": 22, "EVE_36": 23, "EVE_39": 24, 
                       "EVE_41": 25, "EVE_43": 26, "EVE_44": 27, "CUR": 28}
    # returns the predicted result of a previous event, with option to override with new roster
    def prev_sim(self, event, roster=None):
        results = predict(self.pdict, self.popt, self.e_dict[event], roster=roster)
        return results
    # returns player ranking results
    def pr_results(self):
        pr = generate_pr(self.pdict, self.popt)
        return pr
    # returns predicted result of a new event, occuring at current time
    def cur_sim(self, roster):
        results = predict(self.pdict, self.popt, self.e_dict["CUR"], roster=roster)
        indiv = []
        for val in results:
            for i in range(4): indiv.append(round(val/4))
        return indiv
    # computes and returns single event player scores
    def eve_pr(self, event):
        if(event in ["EVE_17", "EVE_46"]):
            return get_scores(event)
        
        ind = min(max(15, self.e_dict[event]+2), self.e_dict["CUR"])
        weights = train(self.pdict, self.popt, e_num=ind, d=[])
        eve_pr = eve_pr_calc(self.pdict, self.popt, self.e_dict[event], weights, d=[])
        return eve_pr

#############################################################################################

# instantiate all_rosters by reading from file
mypath = os.path.dirname(__file__)
txtfile = os.path.join(mypath[:-2], "All_Teamsets.txt")
f = open(txtfile, 'r')
lines = f.read().split("\n")
teams = [x.split("|") for x in lines]
all_rosters = [[x.split(", ") for x in y] for y in teams]

all_rosters = [all_rosters[i] for i in event_select]

all_rosters[27] = [["P020", "P034", "P098", "P233"], ["P068", "P072", "P049", "P062"],
                   ["P028", "P200", "P011", "P035"], ["P055", "P044", "P053", "P210"],
                   ["P223", "P231", "P005", "P015"], ["P001", "P063", "P054", "P079"],
                   ["P022", "P025", "P051", "P225"], ["P050", "P186", "P009", "P141"],
                   ["P227", "P208", "P213", "P232"], ["P042", "P016", "P202", "P203"]]

def fit_func(x,a):
    return 700 * np.tanh(a*x) + 900

class Player:
    def __init__(self):
        self.share = []
        self.stat1 = []
        self.stat2 = []
        self.stat3 = []
        self.stat4 = []

# read in preset scores for uncounted events
def get_scores(eve):
    mypath = os.path.dirname(__file__)
    txtfile = os.path.join(mypath,"scores_uncounted.txt")
    f = open(txtfile, "r")
    text = f.read()
    f.close()
    events = text.split("\n===\n")
    for event in events:
        lines = event.split("\n")
        # select correct event
        if(lines[0] == eve):
            # read scores for each player
            scores = [int(x) for x in lines[2].split(", ")]
            plas = lines[1].split(", ")
            eve_pr = []
            for i in range(len(plas)):
                eve_pr.append([plas[i], scores[i]])
            eve_pr = sorted(eve_pr, key=lambda x: x[1], reverse=True)
            return eve_pr

    return None

def eve_pr_calc(pdict, popt, eve, weights, d=default_d):
    # use either the current event, or for earlier events use event 5
    loc = max(eve, 5) - 5
    plist = []

 
    scores = results[eve*10:(eve+1)*10]

    i = 0
    for team in all_rosters[eve]:
        tots = 0
        ind_scores = []
        for pla in team:
            vec = get_pla_stats(pdict, pla, loc)
            if(pla in non_players):
                vec[0] = pdict["P097"].share[loc]

            vec[0] = fit_func(vec[0] * 4, popt[0]) / 16

            # select variables
            vec = np.delete(vec, d)

            score = max(0, np.matmul(vec, weights))
            # store player scores and total for later manipulation
            tots += score
            ind_scores.append(score)

        # increase very low scores to more reasonable values
        for k in range(len(ind_scores)):
            if(ind_scores[k] < 30):
                tots += 30 - (ind_scores[k]/2+15)
                ind_scores[k] = (ind_scores[k]/2+15)

        # for very early events, smooth the players scores to reflect less confidence
        if(eve < 2):
            for k in range(4):
                ind_scores[k] += 10*(2-eve)
                tots += 10*(2-eve)

        j = 0
        # allocate points for team proportional to each members player score
        for pla in team:
            plist.append([pla, ind_scores[j]*scores[i]/tots])
            j += 1
        i += 1

    # normalize all scores to sum to 10000   
    tot = sum([x[1] for x in plist])
    for i in range(len(plist)):
        plist[i][1] = round(plist[i][1] * 10000 / tot)

    plist = (sorted(plist,key=lambda x: (x[1])))
    plist.reverse()
    
    return plist

def process_data(time_matrix):
    results_t = []

    for k in range(len(time_matrix)):
        eve_times = time_matrix[k]

        # transpose matrix such that each row corresponds to stages, not teams
        eve_times_t = np.transpose(np.copy(eve_times))

        for i in range(9):
            for j in range(10):
                # if stage was not completed, replace time with maximum time for that stage
                # done by checking corresponding row of the transpose matrix
                if eve_times[j][i] == 0:
                    eve_times[j][i] = max(eve_times_t[i]) * 1.1
                    eve_times_t[i][j] = max(eve_times_t[i]) * 1.1
            # if event is outlier event, remove max time when computing average for each stage
            if(outlier[k] == 1):
                avg = (sum(eve_times_t[i]) - max(eve_times_t[i])) / 9
            else: 
                avg = sum(eve_times_t[i]) / 10
            
            # normalize times for the specific stage using the average time
            # higher times yield lower numbers
            for j in range(10):
                eve_times[j][i] = avg / float(eve_times[j][i])
        
        # increase weight of later stages, then get average result for the team over all stages
        for i in range(10):
            eve_times[i][7] = eve_times[i][7] * 1.5
            eve_times[i][8] = eve_times[i][8] * 2
            results_t.append(sum(eve_times[i]) / 10)
    
    # return normalized average scores for all teams
    return results_t


def fill_pdict():
    pdict = {}

    # initialize player objects for each player
    for eve in all_rosters:
        for team in eve:
            for pla in team:
                pdict[pla] = Player()

    # files and variables for substitution data from each contest    
    file_list = ["C4_C9", "C4_C7", "C4_C8", "C4_C1"]
    var_list = ["stat2", "stat1", "stat3", "stat4"]

    mypath = os.path.dirname(__file__)

    # read each substitution data file and assign its respective variable
    for i in range(len(file_list)):
        txtfile = os.path.join(mypath, file_list[i] + ".txt")
        f = open(txtfile, "r")
        text = f.read()
        lines = text.split("\n")
        for line in lines:
            vals = line.split(" ")
            pla = vals[0]
            if(pla not in pdict):
                continue
            # read only the 7th and later events, first 6 are not trained on
            for j in range(len(vals) - 6):
                getattr(pdict[pla], var_list[i]).append(float(vals[j+6]))
        f.close()

    mypath = os.path.dirname(__file__)
    txtfile = os.path.join(mypath,"times.txt")
    f = open(txtfile, "r")
    text = f.read()
    events = text.split("\n===\n")

    # M x N x P matrix, M -> # of events, N -> number of teams (10), P -> number of stages (9)
    time_matrix = []

    # associate team name with one of 10 slots
    name_dict = {"vpode": 0, "njogb": 1, "qimpfcn": 2, "nvfjoae": 3, "beinpa": 4, "nioaeti": 5, "bsro": 6, "mgieroq": 7, "bosrj": 8, "eopjtp": 9}
    
    for event in events:
        # for each event, store list of times for every team
        time_matrix.append([[],[],[],[],[],[],[],[],[],[]])
        stages = event.split("\n\n")
        for stage in stages:
            # read times for each team in each stage
            teams = stage.split("\n")
            for team in teams[1:]:
                words = team.split(" ")
                # if final stage, time is formatted differently
                if("feo mtrp kkgt jwke" in stage):
                    str_time = words[3]
                    if(len(str_time) == 5):
                        value = str_time[0:2]
                        value2 = str_time[3:]
                    else:
                        value = str_time[0]
                        value2 = str_time[2:]
                # otherwise, read using normal format
                else:
                    if(len(words[4]) == 11):
                        str_time = words[4][2:-2]
                    else:
                        str_time = words[4][1:-2]
                    value = str_time[0]
                    value2 = str_time[2:]
                # append time to team's slot in current time matrix
                time = 60 * int(value) + round(float(value2), 2)
                time_matrix[-1][name_dict[words[0]]].append(time)
    
    results_t = process_data(time_matrix)

    ### compute C4 skill for every player after every event
    
    pindex = {}
    index = 0
    l = len(all_rosters)
    w = len(pdict)

    # team player matrix, each row representing a team, each column a player
    # full_matrix[i][j] != 0 means player j is on team i
    full_matrix = np.zeros((l*10, w))

    decay = 1/default_decay
    adj_results = np.copy(results_t)

    for i in range(len(all_rosters)):
        # assign each player an index and populate full_matrix with appearances
        for j, team in enumerate(all_rosters[i]):
            for pla in team:
                if(pla not in pindex):
                    # first appearance, assign index and add target adjustment
                    pindex[pla] = index
                    index += 1
                    adj_results[i*10+j] += 0.025
                    if(pla in new_players):
                        adj_results[i*10+j] += 0.075
                    elif(pla in non_players):
                        adj_results[i*10+j] += 0.125
                # for non_players, always add adjustment
                elif(pla in non_players):
                    adj_results[i*10+j] += 0.1

                # set the column to 1 (time decay reduces impact of older events)
                full_matrix[i*10+j][pindex[pla]] = pow(decay, i)

            adj_results[i*10+j] -= 1
            adj_results[i*10+j] *= pow(decay, i)
        
        # need at least 5 events to get reasonable results
        if(i < 5):
            continue
        
        # select just the assigned portion of full_matrix
        train_matrix = full_matrix[:(i+1)*10, :index]

        # estimate "skill" levels for each player using the constructed sparse matrix
        # apply increasing regularization until results are reasonable (useful for earlier events)
        ans = [1]
        incr = pow(max(train_matrix[-2]) / 2, 2)
        reg = incr
        while(max(ans) > 0.1 or min(ans) < -0.1):
            ident = np.identity(len(train_matrix[0])) * reg
            tmatrix0 = np.transpose(train_matrix)
            fmatrix0 = np.matmul(np.linalg.inv(np.matmul(tmatrix0, train_matrix) + ident), tmatrix0)
            ans = np.matmul(fmatrix0, adj_results[:(i+1)*10])
            reg += incr
        
        # write derived "skill" values for event
        for pla in pindex:
            # if player's first appearance, set previous values to 100 (treated as null)
            if(len(pdict[pla].share) == 0):
                pdict[pla].share.extend([100] * (i - 5))
            
            pdict[pla].share.append(ans[pindex[pla]])


    # initialize substitution variables with nulls if not found
    length = len(pdict["P053"].share)
    for pla in pdict:
        for var in vars(pdict[pla]):
            if(len(getattr(pdict[pla], var)) == 0):
                setattr(pdict[pla], var, [100] * length)

    # using the last 7 events, determine the constants for a tanh fitting of score to points earned
    popt, pcov = sci.curve_fit(fit_func,np.subtract(results_t[e_num*10 - 70:e_num*10],1),results[e_num*10 - 70:e_num*10])

    return pdict, popt

def get_pla_stats(pdict, pla, loc):
    z = pdict[pla]
    s1 = z.share[loc] if z.share[loc] != 100 else pdict[player_subs[pla]].share[loc]
    s2 = z.stat2[loc] if z.stat2[loc] != 100 else pdict[player_subs[pla]].stat2[loc]
    s3 = z.stat3[loc] if z.stat3[loc] != 100 else pdict[player_subs[pla]].stat3[loc]
    s4 = z.stat4[loc] if z.stat4[loc] != 100 else pdict[player_subs[pla]].stat4[loc]
    
    row = [s1, s2, s3, s4, 1]
    # check to ensure no substitution players aren't initialized
    if(max(row) > 10):
        raise Exception("Failed Proxy")
    
    return row

def train(pdict, popt, e_num=-1, d=default_d):
    roster_train = all_rosters[6:]
    results_test = results[60:]

    if(e_num == -1):
        e_num = len(roster_train)
    else:
        e_num = max(e_num - 6, 2)
    
    loc = 0
    matrix_train = []
    for event in roster_train[:e_num]:
        for team in event:
            row = [0,0,0,0,0]
            for pla in team:
                row = np.add(row, get_pla_stats(pdict, pla, loc))
            matrix_train.append(list(row))
        loc += 1
    
    # convert first stat (C4 Skill) into points using previously trained tanh weights
    for row in matrix_train:
        row[0] = fit_func(row[0], popt[0]) / 4
    
    # select variables
    matrix_train = np.delete(matrix_train, d, axis=1)

    # compute weight vector associating stats with results
    tmatrix = np.transpose(matrix_train)
    fmatrix = np.matmul(np.linalg.inv(np.matmul(tmatrix, matrix_train)), tmatrix)
    ans = np.matmul(fmatrix, results_test[0:e_num*10])

    return ans

def predict(pdict, popt, e_num, d=default_d, roster=None):
    # get weights by training with all data prior to event
    weights = train(pdict, popt, e_num, d)

    if(roster is not None):
        predict_roster = roster
    else:
        predict_roster = all_rosters[e_num]
    
    # build the stat matrix, sum of player stats for each team
    inputs_matrix = []
    for team in predict_roster:
        row = [0,0,0,0,0]
        for pla in team:
            row = np.add(row, get_pla_stats(pdict, pla, e_num-6))
        inputs_matrix.append(list(row))
    
    # convert first stat (C4 Skill) into points using previously trained tanh weights
    for row in inputs_matrix:
        row[0] = fit_func(row[0], popt[0]) / 4
    
    inputs_matrix = np.delete(inputs_matrix, d, axis=1)
    
    result = np.matmul(inputs_matrix, weights)

    # normalize result to 10000
    norm = 10000 / sum(result)
    result = result * norm
    result = result.round()

    # return results
    return result

def generate_pr(pdict, popt, d=default_d):
    # get weights by training on all data
    weights = train(pdict, popt, d=d)

    matrix_pr = []
    pr = []

    for pla in pdict:
        # if player hasn't appeared in enough events, do not include
        if(pdict[pla].share[-1] == 100 or (pdict[pla].stat4[-1] == 100 and pdict[pla].stat3[-1] == 100)):
            continue
        
        matrix_pr.append(get_pla_stats(pdict, pla, len(pdict[pla].share)-1))
        pr.append([pla])
    
    # convert first stat (C4 Skill) into points using previously trained tanh weights
    # multiply by 4 initially as weights are calibrated for a full team score
    for row in matrix_pr:
        row[0] = fit_func(row[0]*4, popt[0]) / 16
    
    # compute scores for each player using precomputed weight matrix
    matrix_pr = np.delete(matrix_pr, d, axis=1)
    scores = np.matmul(matrix_pr, weights)
    for i in range(len(pr)):
        pr[i].append(float(round(scores[i])))

    pr = (sorted(pr,key=lambda x: (x[1])))
    pr.reverse()
    
    # read player frequencies from current players file
    f = open(os.path.join(os.path.split(os.path.dirname(__file__))[0], "current_list.txt"))
    text = f.read()
    lines = text.split("\n")
    pfreq_dict = {}
    for line in lines:
        values = line.split(" ")
        pfreq_dict[values[0]] = values[1]
    
    # remove non-current players or negative outlier players
    ind = 0
    while(ind < len(pr)):
        if(pr[ind][0] not in pfreq_dict.keys() or pr[ind][1] < 0):
            del pr[ind]
        else:
            ind += 1

    # normalize all scores based on player frequency, to the average score
    avg_scores = 0
    total = 0
    for player in pr:
        avg_scores += player[1] * float(pfreq_dict[player[0]])
        total += float(pfreq_dict[player[0]])
    avg_scores /= total
    for player in pr:
        player[1] = round(player[1] * (250 / avg_scores))
    
    # return pr
    return pr

if __name__ == "__main__":

    cla = C4_class()
    pdict = cla.pdict
    popt = cla.popt

    perfList = []
    for key in cla.e_dict:
        if(key != "CUR"):
            val = cla.eve_pr(key)
            for value in val:
                perfList.append([value[1], value[0] + key[-3:]])
    perfList = sorted(perfList, reverse=True)

    val = cla.eve_pr("EVE_39")

    # if current event, return pr
    if(cla.e_dict["CUR"] == e_num):
        pr = generate_pr(pdict, popt)
        string = ""
        for player in pr:
            string = string + (player[0] + "\t" + str(player[1]) + "\n")
        
        check = 0
    else:
        test = predict(pdict, popt, e_num)

        # check against true results for mse accuracy
        results_test = results[e_num*10:e_num*10+10]
        error = 0
        errorm = []
        for j in range(10):
            error += np.power((test[j] - results_test[j]), 2)
            errorm.append(test[j] - results_test[j])
        error = error / 10
        error = np.sqrt(error)

        # evaluate mse accuracy of another prediction method for comparison
        error2 = 0
        error2m = []
        pr_preds_range = pr_preds[(e_num-8)*10:(e_num-7)*10]
        for j in range(10):
            error2 += np.power((pr_preds_range[j] - results_test[j]), 2)
            error2m.append(pr_preds_range[j] - results_test[j])
        error2 = error2 / 10
        error2 = np.sqrt(error2)

        # evaluate mse accuracy of naive prediction method for comparison
        error3 = 0
        equal_values = [990,990,990,990,990,990,990,990,990,990]
        for j in range(10):
            error3 += np.power((equal_values[j] - results_test[j]), 2)
        error3 = error3 / 10
        error3 = np.sqrt(error3)

        errom = [int(x) for x in errorm]
        errom2 = [pow(x,2) for x in errom]
        error2m2 = [pow(x,2) for x in error2m]

        print(error)
        print(error2)
        print(error3)

        check = 0