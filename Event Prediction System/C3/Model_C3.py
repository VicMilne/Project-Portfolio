import numpy as np
import os.path

##############################################################################################

# Consistently Updated Globals and Classes

default_decay = 0.95
default_d = [1,3,4]

point_average = 800

e_num = 19

# numbered events where this contest appeared
event_select = [0,1,3,4,5,6,7,8,11,13,18,21,26,29,34,35,37,38,40]

# team by team results for all events
results = [1040,1420,1452,1144,704,760,1192,1128,1180,1104,
        1760,1076,192,1208,496,620,1468,1556,1292,1648,
        1136,1468,1076,1000,780,1224,1404,512,1564,1028,
        1100,1800,1456,1356,1068,760,688,600,704,1204,
        712,676,1660,1508,1052,1152,892,1064,912,912,
        1192,1017,1548,1408,1592,1252,1200,1444,760,1180,
        1048,1676,1148,944,1540,1248,824,288,912,1292,
        492,832,1832,1280,1780,912,1668,920,720,1164,
        871,1268,1931,620,98,1055,1094,409,1130,782,
        1557,1521,800,1126,1357,597,995,778,1305,1041,
        606,384,220,490,771,923,610,1263,719,796,
        1648,1399,1525,542,1224,696,1002,473,1030,1384,
        1392,1414,706,1753,1801,1117,1115,804,915,1387,
        714,1065,2163,951,1115,1339,853,734,460,338,
        802,1533,919,828,934,256,1611,949,432,1584,
        990,791,689,981,695,1401,1641,1286,1657,188,
        500,458,720,583,269,401,446,750,623,1124,
        562,959,545,393,1185,1074,645,581,590,1110,
        720,945,1103,897,770,689,524,428,561,1194]

# baseline predictions to compare against
pr_preds = [1000,1000,999,1000,999,1000,999,1000,999,999,
            1369,1209,391,1259,1206,649,948,1098,1217,654,
            1120,1120,1120,1120,1120,1120,1120,1120,1120,1120,
            1120,1120,1120,1120,1120,1120,1120,1120,1120,1120,
            1119,865,980,1323,1447,611,1354,425,1255,1358,
            853,1550,1201,1305,528,598,1189,1063,669,1289,
            1573,1769,1049,962,1262,586,1176,926,751,1231,
            1115,760,784,967,1287,1233,1152,1210,1833,450,
            1013,1228,1453,488,281,1114,1192,1367,1070,1205,
            419,1153,522,577,944,829,1017,1000,880,1060,
            1306,732,1157,1064,770,1558,1471,204,561,1121
            ]

player_subs = {"P135": "P035", "P004": "P062", "P047": "P057", "P157": "P097", "P099": "P097", "P097": "P097",
"P046": "P001", "P016": "P035", "P026": "P118", "P134": "P032", "P130": "P024", "P056": "P010", "P006": "P017",
"P034": "P045", "P126": "P035", "P008": "P096", "P041": "P029", "P021": "P043", "P136": "P127", "P024": "P035",
"P052": "P024", "P028": "P074", "P005": "P016", "P002": "P011", "P030": "P026", "P009": "P047",
"P063": "P035", "P133": "P038", "P020": "P046", "P059": "P061", "P060": "P035", "P078": "P053", "P050": "P011",
"P124": "P044", "P064": "P035", "P000": "P062", "P125": "P006", "P017": "P024", "P137": "P060", "P039": "P144",
"P073": "P024", "P065": "P001", "P015": "P024", "P025": "P028", "P040": "P032", "P027": "P010",
"P075": "P024", "P061": "P059", "P132": "P017", "P115": "P118", "P096": "P127", "P086": "P033", "P120": "P017",
"P121": "P026", "P037": "P043", "P106": "P043", "P071": "P026", "P150": "P097", "P170": "P097",
"P171": "P097", "P172": "P097", "P167": "P096", "P168": "P006", "P117": "P057", "P107": "P035",
"P152": "P097", "P148": "P040", "P143": "P035", "P141": "P024", "P145": "P053", "P071": "P026",
"P153": "P097", "P154": "P097", "P147": "P032", "P155": "P096", "P151": "P097", "P149": "P049",
"P146": "P024", "P105": "P096", "P129": "P033", "P007": "P097", "P139": "P024", "P079": "P051", 
"P003": "P097", "P156": "P096", "P138": "P026", "P077": "P030", "P140": "P097", "P142": "P042",
"P069": "P154", "P197": "P026", "P198": "P063", "P199": "P043", "P118": "P030", "P098": "P047",
"P068": "P042", "P076": "P075", "P109": "P097", "P023": "P025", "P055": "P029", "P177": "P097",
"P202": "P027", "P203": "P064", "P205": "P097", "P204": "P052", "P178": "P047", "P179": "P067",
"P210": "P063", "P213": "P030", "P220": "P032", "P221": "P064", "P219": "P028", "P202": "P141", 
"P222": "P012", "P215": "P097", "P216": "P096", "P217": "P096", "P218": "P097", "P223": "P011",
"P224": "P060", "P225": "P077", "P105": "P096", "P226": "P051", "P124": "P011", "P227": "P012",
"P231": "P010", "P142": "P223", "P214": "P011"
}

new_players = {"P117", "P107", "P105", "P135", "P136", "P146", "P071", "P198", "P199", 
               "P203", "P077", "P221", "P225", "P105"}
non_players = {"P158", "P161", "P160", "P159", "P162", "P164", "P165", "P163", 
               "P157", "P166", "P169", "P120", "P170", "P172", "P171", "P154", "P139", 
               "P177", "P152", "P197", "P205", "P099", "P146", "P218", "P216", "P217", "P215"}

class C3_class:
    def __init__(self):
        self.pdict = fill_pdict()
        self.e_dict = {"EVE_1": 0, "EVE_2": 1, "EVE_4": 2, "EVE_5": 3, "EVE_6": 4, "EVE_7": 5,
                       "EVE_8": 6, "EVE_9": 7, "EVE_12": 8, "EVE_14": 9, "EVE_23": 11, "EVE_28": 12, 
                       "EVE_31": 13, "EVE_36": 14, "EVE_37": 15, "EVE_39": 16, "EVE_40": 17, "EVE_42": 18, 
                       "CUR": 19}
    # returns the predicted result of a previous event, with option to override with new roster
    def prev_sim(self, event, roster=None):
        results = predict(self.pdict, self.e_dict[event], roster=roster)
        return results
    # returns player ranking results
    def pr_results(self):
        pr = generate_pr(self.pdict)
        return pr
    # returns predicted result of a new event, occuring at current time
    def cur_sim(self, roster):
        results = predict(self.pdict, self.e_dict["CUR"], roster=roster)
        indiv = []
        for val in results:
            for i in range(4): indiv.append(round(val/4))
        return indiv
    # computes and returns single event player scores
    def eve_pr(self, event):
        if(event == "EVE_32"):
            return nceve1
        
        eve_pr = eve_pr_calc(self.pdict, self.e_dict[event])
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

nceve1 = [["P179", 475], ["P178", 475], ["P180", 475], ["P181", 475], ["P193", 425], ["P191", 425],
           ["P194", 425], ["P192", 425], ["P187", 350], ["P000", 350], ["P188", 350], ["P056", 350],
           ["P042", 263], ["P060", 263], ["P033", 263], ["P189", 263], ["P142", 225], ["P190", 225],
           ["P027", 225], ["P015", 225], ["P020", 188], ["P049", 188], ["P009", 188], ["P141", 188],
           ["P022", 163], ["P040", 163], ["P019", 163], ["P183", 163], ["P195", 150], ["P062", 150], 
           ["P196", 150], ["P005", 150], ["P050", 138], ["P184", 138], ["P185", 138], ["P186", 138],
           ["P034", 125], ["P053", 125], ["P051", 125], ["P182", 125]]

class Player:
    def __init__(self):
        self.share = []
        self.stat1 = []
        self.stat2 = []
        self.stat3 = []
        self.stat4 = []
        self.stat5 = []

def eve_pr_calc(pdict, eve, d=default_d):
    weights = train(pdict)
    # use either the current event, or for earlier events use event 6
    loc = max(eve, 6) - 5
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
            
            if(e_num <= 5 and 5 not in d):
                d.append(5)
    
            # select variables
            vec = np.delete(vec, d)

            score = max(np.matmul(vec, weights), 0)
            # store player scores and total for later manipulation
            tots += score
            ind_scores.append(score)
        
        # increase very low scores to more reasonable values
        for k in range(len(ind_scores)):
            if(ind_scores[k] < 120): 
                tots += 120 - (ind_scores[k]/2+60)
                ind_scores[k] = (ind_scores[k]/2+60)
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

# normalize all data by event, removing outlier results if necessary
def process_data(results):
    outliers = {1: [5,4,2], 8: [9,7,4], 11: [7,5], 13: [9], 16: [9]}
    adj_results = np.copy(results).astype(float)
    for i in range(round(len(results)/10)):
        # determine the average of the non-outlier results
        e_results = list(adj_results[i*10:(i+1)*10])
        divisor = 10
        if(i in outliers):
            divisor -= len(outliers[i])
            for val in outliers[i]:
                del e_results[val]
        aver = sum(e_results) / divisor

        # divide the 10 results by the average
        for j in range(10):
            adj_results[i*10 + j] = adj_results[i*10 + j] / aver - 1
    return adj_results

def fill_pdict():
    pdict = {}
    # normalize data
    adj_results = process_data(results)
    
    # initialize player objects for each player
    for eve in all_rosters:
        for team in eve:
            for pla in team:
                if(pla not in pdict):
                    pdict[pla] = Player()
    
    # files and variables for substitution data from each contest
    file_list = ["C3_C9", "C3_C7", "C3_C8", "C3_C1", "C3_CA"]
    var_list = ["stat2", "stat1", "stat3", "stat4", "stat5"]

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
    
    ### compute C3 skill for every player after every event
    
    pindex = {}
    index = 0
    l = len(all_rosters)
    w = len(pdict)

    # team player matrix, each row representing a team, each column a player
    # full_matrix[i][j] != 0 means player j is on team i
    full_matrix = np.zeros((l*10, w))

    decay = 1/default_decay

    for i in range(len(all_rosters)):
        # assign each player an index and populate full_matrix with appearances
        for j, team in enumerate(all_rosters[i]):
            for pla in team:
                if(pla not in pindex):
                    # first appearance, assign index and add target adjustment
                    pindex[pla] = index
                    index += 1
                    adj_results[i*10+j] += 0.05
                    if(pla in new_players):
                        adj_results[i*10+j] += 0.05 

                # set the column to 1 (time decay reduces impact of older events)
                full_matrix[i*10+j][pindex[pla]] = pow(decay, i) 

            adj_results[i*10+j] *= pow(decay, i)
        
        # need at least 5 events to get reasonable results
        if(i < 5):
            continue
        
        # select just the assigned portion of full_matrix
        train_matrix = full_matrix[:(i+1)*10, :index]

        # estimate "skill" levels for each player using the constructed sparse matrix
        # apply increasing regularization until results are reasonable (useful for earlier events)
        ans = [1]
        incr = pow(max(train_matrix[-2]) / 3, 2)
        reg = incr
        while(max(ans) > 0.175 or min(ans) < -0.175):
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
            if(pla in non_players):
                pdict[pla].share.append(-0.1)
            else:
                pdict[pla].share.append(ans[pindex[pla]])

    # initialize substitution variable with nulls if not found
    length = len(pdict["P053"].share)
    for pla in pdict:
        for var in vars(pdict[pla]):
            if(len(getattr(pdict[pla], var)) == 0):
                setattr(pdict[pla], var, [100] * length)
    
    return pdict

def get_pla_stats(pdict, pla, loc):
    z = pdict[pla]
    s1 = z.share[loc] if z.share[loc] != 100 else pdict[player_subs[pla]].share[loc] - 0.05
    s2 = z.stat2[loc] if z.stat2[loc] != 100 else pdict[player_subs[pla]].stat2[loc]
    s3 = z.stat3[loc] if z.stat3[loc] != 100 else pdict[player_subs[pla]].stat3[loc]
    s4 = z.stat4[loc] if z.stat4[loc] != 100 else pdict[player_subs[pla]].stat4[loc]
    s5 = z.stat1[loc] if z.stat1[loc] != 100 else pdict[player_subs[pla]].stat1[loc]
    # use default 0 value if early event or player substitution also not initialized
    if(z.stat5[loc] == 100 and (loc <= 4 or pla not in player_subs or pdict[player_subs[pla]].stat5[loc] == 100)):
        s6 = 0
    elif(z.stat5[loc] == 100):
        s6 = pdict[player_subs[pla]].stat5[loc]
    else:
        s6 = z.stat5[loc]
    
    row = [s1+0.25, s2, s3, s4, s5, s6, 1]
    # check to ensure no substitution players aren't initialized
    if(max(row) > 10):
        raise Exception("Wrong Proxy")
    
    return row

def train(pdict, e_num=-1, d=default_d):
    # define just the training events
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
            row = [0,0,0,0,0,0,0]
            for pla in team:
                row = np.add(row, get_pla_stats(pdict, pla, loc))
            matrix_train.append(list(row))
        loc += 1

    # remove additional variable if e_num <= 5
    if(e_num <= 5 and 5 not in d):
        d.append(5)
    
    # select variables
    matrix_train = np.delete(matrix_train, d, axis=1)

    # compute weight vector associating stats with results
    tmatrix = np.transpose(matrix_train)
    fmatrix = np.matmul(np.linalg.inv(np.matmul(tmatrix, matrix_train)), tmatrix)
    ans = np.matmul(fmatrix, results_test[0:e_num*10])

    return ans

def predict(pdict, e_num, d=default_d, roster=None):
    # get weights by training with all data prior to event
    weights = train(pdict, e_num, d)

    if(roster is not None):
        predict_roster = roster
    else:
        predict_roster = all_rosters[e_num]

    # build the stat matrix, sum of player stats for each team
    inputs_matrix = []
    for team in predict_roster:
        row = [0,0,0,0,0,0,0]
        for pla in team:
            row = np.add(row, get_pla_stats(pdict, pla, e_num-6))
        inputs_matrix.append(list(row))
    
    # if early event, add stat 5 to removed stats
    if(e_num-6 <= 5 and 5 not in d):
        d.append(5)
    
    inputs_matrix = np.delete(inputs_matrix, d, axis=1)

    result = np.matmul(inputs_matrix, weights)
    # normalize amount of points
    point_pool = point_average*10
    for team in predict_roster:
        for pla in team:
            # if players are present that reduce talent level, lower normalization factor
            if(pla in non_players):
                point_pool -= 200

    norm = point_pool / sum(result)
    result = result * norm
    result = result.round()
    # return results
    return result

def generate_pr(pdict, d=default_d):
    # get weights by training on all data
    weights = train(pdict, d=d)

    matrix_pr = []
    pr = []

    for pla in pdict:
        # if player hasn't appeared in enough events, do not include
        if(pdict[pla].share[-1] == 100 or (pdict[pla].stat4[-1] == 100 and pdict[pla].stat3[-1] == 100)):
            continue
        
        matrix_pr.append(get_pla_stats(pdict, pla, len(pdict[pla].share)-1))
        pr.append([pla])
    
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

    cla = C3_class()
    pdict = cla.pdict

    # eve_pr test
    val = cla.eve_pr("EVE_40")

    # if current event, return pr
    if(cla.e_dict["CUR"] == e_num):
        pr = generate_pr(pdict)
        string = ""
        for player in pr:
            string = string + (player[0] + "\t" + str(player[1]) + "\n")
        
        check = 0
    else:
        test = predict(pdict, e_num)

        # check against true results for mse accuracy
        results_test = results[e_num*10:e_num*10+10]
        error = 0
        errorm = []
        scale = sum(test) / sum(results_test)
        for j in range(10):
            error += np.power((test[j]/scale - results_test[j]), 2)
            errorm.append(test[j] - results_test[j])
        error = error / 10
        error = np.sqrt(error)

        # evaluate mse accuracy of another prediction method for comparison
        error2 = 0
        error2m = []
        pr_preds_range = pr_preds[(e_num-8)*10:(e_num-7)*10]
        scale = sum(pr_preds_range) / sum(results_test)
        for j in range(10):
            error2 += np.power((pr_preds_range[j]/scale - results_test[j]), 2)
            error2m.append(pr_preds_range[j] - results_test[j])
        error2 = error2 / 10
        error2 = np.sqrt(error2)

        # evaluate mse accuracy of naive prediction method for comparison
        error3 = 0
        equal_values = [1120,1120,1120,1120,1120,1120,1120,1120,1120,1120]
        for j in range(10):
            error3 += np.power((equal_values[j] - results_test[j]), 2)
        error3 = error3 / 10
        error3 = np.sqrt(error3)

        errom = [int(x) for x in errorm]
        errom2 = [pow(x,2) for x in errom]
        error2m2 = [pow(x,2) for x in error2m]


        print(error)
        print(error2)
        check = 0