import numpy as np
import os.path
from time import perf_counter

########################################################################################

# Consistently Updated Globals and Classes

default_decay = 0.9
default_d = [0, 5, 6, 7, 10, 13]

eve_num = 16

# numbered events where this contest appeared
event_select = [7,9,10,13,14,15,16,17,19,22,23,26,27,28,33,35,37,40,41]

# team by team results for all events
results = [1570, 795, 560, 910, 1380, 1980, 1200, 15, 910, 935,
    940, 1210, 750, 185, 915, 1225, 1085, 1685, 1035, 1020,
    1000, 1045, 1335, 1395, 795, 1775, 1010, 530, 810, 780,
    910, 645, 1600, 1015, 810, 970, 925, 1775, 985, 790,
    1380, 680, 925, 545, 1160, 1290, 1345, 955, 995, 955,
    825, 985, 1380, 825, 970, 1145, 1410, 1220, 1190, 705,
    1190, 1980, 1190, 545, 1335, 1540, 370, 495, 750, 1000,
    985, 1010, 750, 1365, 515, 1205, 1130, 1000, 1600, 880,
    1395, 620, 1600, 940, 1380, 355, 765, 1380, 1365, 910,
    940, 1015, 1290, 1215, 500, 720, 1205, 1525, 925, 865,
    1255, 445, 1760, 1455, 750, 1160, 605, 725, 1145, 590,
    1540, 1380, 630, 515, 1390, 1775, 1000, 935, 325, 515,
    1015, 1365, 840, 1015, 1585, 735, 1015, 105, 1100, 1980,
    1775, 1320, 1290, 1380, 1175, 835, 910, 1090, 545, 75,
    760, 720, 1395, 865, 1240, 1160, 955, 690, 1130, 1030,
    1335, 1000, 1130, 1160, 910, 995, 1340, 340, 1055, 560,
    970, 1260, 1190, 780, 15, 1160, 1275, 1380, 1110, 685,
    1115, 865, 660, 720, 1145, 955, 1145, 1130, 970, 1540,
    1220, 515, 1020, 850, 1775, 780, 705, 1980, 705, 620]

# baseline predictions to compare against
pr_preds = [988, 690, 721, 1556, 1013, 1595, 622, 1387, 1314, 413, 
    1236, 890, 1627, 947, 1613, 584, 1114, 677, 1151, 451,
    1139, 599, 1364, 363, 750, 1347, 1239, 1439, 888, 1158,
    1132, 1162, 1080, 1192, 1206, 1263, 1066, 191, 1117, 1001,
    1232, 840, 707, 827, 1527, 1145, 1434, 358, 1065, 1305,
    1167, 890, 865, 662, 1004, 428, 1482, 1298, 1608, 1036,
    895, 1283, 1650, 1358, 1323, 671, 635, 581, 1202, 920,
    1265, 1055, 1037, 1238, 569, 891, 1378, 1265, 676, 681,
    1183, 735, 1347, 766, 865, 1564, 1416, 1282, 353, 753,
    965, 1318, 964, 807, 1180, 1069, 1093, 251, 858, 1760,
    993, 983, 1472, 884, 1179, 1129, 420, 1243, 1558, 405,
    663, 430, 1168, 1396, 1395, 1099, 335, 1279, 1664, 964,
    978, 815, 1754, 1180, 1142, 1252, 817, 1053, 1125, 299,
    938, 842, 1141, 1076, 231, 1522, 1296, 1383, 1228, 759,
    902, 903, 1503, 1030, 425, 982, 1433, 567, 1188, 1372,
    1040, 1040, 1040, 1040, 1040, 1040, 1040, 1040, 1040, 1040]

reduced_events = [18, 20]

player_subs = {"P135": "P084", "P047": "P061", "P099": "P127", "P096": "P127", "P097": "P127",
                  "P144": "P011", "P017": "P051", "P023": "P044", "P038": "P035", "P055": "P062",
                  "P004": "P011", "P064": "P084", "P067": "P044", "P016": "P017", "P046": "P000", 
                  "P078": "P038", "P026": "P127", "P056": "P051", "P006": "P084", "P024": "P051",
                  "P034": "P074", "P043": "P006", "P126": "P055", "P008": "P032", "P021": "P000",
                  "P136": "P127", "P012": "P084", "P052": "P048", "P028": "P062", "P005": "P033", 
                  "P002": "P006", "P063": "P017", "P125": "P011", "P030": "P061", "P009": "P061", 
                  "P133": "P033", "P020": "P074", "P124": "P061", "P050": "P048", "P039": "P048",
                  "P059": "P000", "P073": "P026", "P015": "P056", "P137": "P050", "P065": "P054",
                  "P040": "P044", "P027": "P005", "P041": "P023", "P075": "P026", "P115": "P012", "P076": "P099",
                  "P077": "P099", "P068": "P036", "P069": "P038", "P070": "P023", "P071": "P026",
                  "P072": "P006", "P143": "P051", "P141": "P073", "P200": "P058", "P025": "P032",
                  "P118": "P060", "P098": "P028", "P094": "P075", "P205": "P097", "P179": "P073", "P204": "P049",
                  "P178": "P028", "P203": "P051", "P202": "P141", "P145": "P065", "P153": "P077", "P213": "P002",
                  "P215": "P097", "P218": "P143", "P217": "P012", "P216": "P027", "P210": "P030", "P219": "P055",
                  "P220": "P049", "P221": "P056", "P149": "P040", "P222": "P016", "P000": "P028", "P223": "P042",
                  "P225": "P202", "P224": "P044", "P227": "P098", "P231": "P060", "P105": "P015", "P226": "P027",
                  "P139": "P051", "P140": "P073", "P214": "P022", "P142": "P001", "P022": "P001"
                  }

class C2_class:
    def __init__(self):
        self.pdict = fill_pdict()
        self.eve_str = compute_eve_str(self.pdict)
        self.e_dict = {"EVE_1": -6, "EVE_3": -5, "EVE_6": -4, "EVE_8": -3, "EVE_10": -2, "EVE_11": -1,
                       "EVE_14": 0, "EVE_15": 1, "EVE_16": 2, "EVE_18": 3, "EVE_19": 4, "EVE_21": 5, "EVE_24": 6, 
                       "EVE_25": 7, "EVE_28": 8, "EVE_29": 9, "EVE_30": 10, "EVE_35": 11, "EVE_37": 12, 
                       "EVE_39": 13, "EVE_42": 14, "EVE_43": 15, "CUR": 16}
        self.nc_eves = {"EVE_2": 4, "EVE_7": 5, "EVE_12": 5, "EVE_17": 7, "EVE_23": 8, "EVE_31": 11, 
                        "EVE_36": 13, "EVE_40": 15, "EVE_44": 16}
    # returns the predicted result of a previous event, with option to override with new roster
    def prev_sim(self, event, roster=None):
        results = predict(self.pdict, self.e_dict[event], self.eve_str, roster=roster)
        return results[0]
    # returns player ranking results
    def pr_results(self):
        pr = generate_pr(self.pdict, self.eve_str)
        return pr
    # returns predicted result of a new event, occuring at current time
    def cur_sim(self, roster):
        _, results = predict(self.pdict, self.e_dict["CUR"], self.eve_str, roster=roster)
        return results
    # computes and returns single event player scores
    def eve_pr(self, event):
        if(event in self.nc_eves):
            # need the uncounted data for the above events
            dict_nc = fill_pdict(True, event)

            weights = train(self.pdict, self.eve_str, self.nc_eves[event], d=[2,3,6,7,9,12,13])
            eval_pr = eve_pr_calc(dict_nc, -6, weights, d=[2,3,6,7,9,12,13])
            # uncounted events have more variance, reduce std of scores
            for val in eval_pr:
                val[1] = int((val[1] - 250)*0.85 + 250)
            return eval_pr

        ind = min(max(4, self.e_dict[event]+2), self.e_dict["CUR"])
        weights = train(self.pdict, self.eve_str, eve_num=ind, d=[1, 4, 5, 7, 10, 13])
        eval_pr = eve_pr_calc(self.pdict, self.e_dict[event], weights, d=[1, 4, 5, 7, 10, 13])
        return eval_pr

#############################################################################################

# instantiate all_rosters by reading from file
mypath = os.path.dirname(__file__)
txtfile = os.path.join(mypath[:-2], "All_Teamsets.txt")
f = open(txtfile, 'r')
lines = f.read().split("\n")
teams = [x.split("|") for x in lines]
all_rosters = [[x.split(", ") for x in y] for y in teams]

# select rosters
all_rosters = [all_rosters[i] for i in event_select]

# player class storing stats events lists for each stat
class Player:
    def __init__(self, num_eve):
        self.stat1 = [-1] * num_eve
        self.stat3 = [-1] * num_eve
        self.stat4 = [-1] * num_eve
        self.tstat1 = [-1] * num_eve
        self.stat2 = [-1] * num_eve
        self.tstat2 = [-1] * num_eve
        self.stat5 = [-1] * num_eve
        self.stat6 = [-1] * num_eve

        self.num_last7 = [-1] * num_eve

# player class storing stats going into a particular event
class eventPlayer:
    def __init__(self):
        self.stat1 = 0
        self.stat12 = 0
        self.tstat1 = 0
        self.tstat12 = 0
        self.stat2 = 0
        self.stat22 = 0
        self.tstat2 = 0
        self.tstat22 = 0
        self.stat3 = 0
        self.stat32 = 0
        self.stat4 = 0
        self.stat42 = 0
        self.stat5 = 0
        self.stat52 = 0

# calculating pr for a particular event
def eve_pr_calc(pdict, eve, weights, d=default_d):
    loc = eve + 6
    matrix = []
    plist = []

    # create stats matrix of every player in the particular event
    for pla in pdict:
        if(pdict[pla].stat1[loc] == -1):
            continue
        z = pdict[pla]
        a = 1

        # if edge condition, buff player results
        if(z.tstat2[loc] < 15 and z.stat3[loc] > 4):
            a = 2.25 - min(0.75, z.stat5[loc])/0.75

        matrix.append([z.stat1[loc]*a, pow(z.stat1[loc]*a, 2), z.stat2[loc]*a, pow(z.stat2[loc]*a, 2), z.tstat1[loc]*a, 
                       pow(z.tstat1[loc]*a, 2), z.tstat2[loc]*a, pow(z.tstat2[loc]*a, 2), (4+z.stat3[loc])/2, pow((4+z.stat3[loc])/a/2, 2),  
                z.stat4[loc], pow(z.stat4[loc], 2), z.stat5[loc], pow(z.stat5[loc], 2), 1, 1])
        plist.append([pla])

    
    matrix = np.delete(matrix, d, axis=1)

    # compute scores using weights vector
    vals = np.matmul(matrix, weights)

    # formula for adjusting all event scores to be above 0, maintains relative positions
    x = min(vals)
    if(x < 0):
        m = 30 / (30 - x)
        b = 30 - 30*m
        for i in range(len(vals)):
            if(vals[i] < 30):
                vals[i] = vals[i]*m + b

    # normalize and round, then pair scores with players
    f = 10000 / sum(vals)
    for i in range(len(plist)):
        plist[i].append(round(vals[i]*f))

    # sort plist by score, descending
    plist = (sorted(plist,key=lambda x: (x[1])))
    plist.reverse()

    return plist

# computes time decayed averages for all stats up to time loc
# performs outlier analysis
def build_player(pdict, loc, pla, eve_str, decay):
    inc = loc
    depth = 0
    div = 0
    outlier1 = [[],[],[],[],[],[],[]]
    outlier2 = []
    event_player = eventPlayer()

    while(inc >= 0):
        # increase decay for reduced_events or stat6 events
        revert = False
        if(inc-1 in reduced_events or pdict[pla].stat6[inc] == 1):
            depth = depth + 7
            revert = True
        
        value = pdict[pla].stat1[inc]

        # if player stats for event don't exist (== -1) skip
        if value != -1:
            # adjustment for event comp
            value *= eve_str[inc]/eve_str[loc+1]

            event_player.stat1 += value * pow(decay, depth)
            event_player.stat12 += pow(value,2) * pow(decay, depth)
            div += pow(decay,depth)
            if(depth < 8):
                outlier1[0].append(value * pow(decay, depth))
                outlier2.append(pow(decay, depth))

            value = pdict[pla].tstat1[inc]
            value = value + (eve_str[inc]/eve_str[loc+1] - 1) * pdict[pla].stat1[inc]
            event_player.tstat1 += value * pow(decay, depth)
            event_player.tstat12 += pow(value,2) * pow(decay, depth)
            if(depth < 8):
                outlier1[1].append(value * pow(decay, depth))

            value = pdict[pla].stat2[inc]
            value = value * eve_str[inc]/eve_str[loc+1]
            event_player.stat2 += value * pow(decay, depth)
            event_player.stat22 += pow(value,2) * pow(decay, depth)
            if(depth < 8):
                outlier1[2].append(value * pow(decay, depth))
                
            value = pdict[pla].tstat2[inc]
            value = value + (eve_str[inc]/eve_str[loc+1] - 1) * pdict[pla].stat2[inc]
            event_player.tstat2 += value * pow(decay, depth)
            event_player.tstat22 += pow(value,2) * pow(decay, depth)
            if(depth < 8):
                outlier1[3].append(value * pow(decay, depth))
        
            value = pdict[pla].stat3[inc]
            value = value * (((eve_str[inc]/eve_str[loc+1]) - 1) / 4 + 1)
            event_player.stat3 += value * pow(decay, depth)
            event_player.stat32 += pow(value,2) * pow(decay, depth)
            if(depth < 8):
                outlier1[4].append(value * pow(decay, depth))

            value = pdict[pla].stat4[inc]
            value = value * eve_str[inc]/eve_str[loc+1]
            event_player.stat4 += value * pow(decay, depth)
            event_player.stat42 += pow(value,2) * pow(decay, depth)
            if(depth < 8):
                outlier1[5].append(value * pow(decay, depth))

            value = pdict[pla].stat5[inc]
            value = value * eve_str[inc]/eve_str[loc+1]
            event_player.stat5 += value * pow(decay, depth)
            event_player.stat52 += pow(value,2) * pow(decay, depth)
            if(depth < 8):
                outlier1[6].append(value * pow(decay, depth))
       
        inc -= 1
        depth += 1

        if(revert):
            depth -= 7
    
    # do outlier analysis if more than 3 events
    if(len(outlier2) > 3):
        # recompute original values using decayed value and amount of decay
        full_values = [outlier1[2][i] / outlier2[i] for i in range(len(outlier2))]
        mean = np.mean(full_values)
        std = np.std(full_values)
        
        for k in range(len(full_values)):
            # if stat2 is outlier 
            if(abs((full_values[k] - mean)/std) > 1.5):
                # iterate through variables in order defined in class
                for i, var in enumerate(vars(event_player)):
                    # reduce outlier's effect by 50%, for squared values need to compute square first
                    if(i % 2 == 0):
                        setattr(event_player, var, vars(event_player)[var] - outlier1[i//2][k] * 0.5)
                    else:
                        setattr(event_player, var, vars(event_player)[var] - pow(outlier1[i//2][k], 2) * 0.5 / outlier2[k])
                div -= outlier2[k] * 0.5

    # divide all stat totals by divisor to determine weighted average        
    for var in vars(event_player):
        setattr(event_player, var, vars(event_player)[var] / div)

    return event_player

# logic for established versus new players, calls build_player() to get stats (including subs)
def compute_eve_pla(pdict, loc, pla, eve_str, decay, ispr):
    # if player not initialized or (haven't played many recent events and doing predictions)
    if(pla not in pdict or (pdict[pla].num_last7[loc+1] < 2 and not ispr)):
        # determine player substitution's values
        pla_b = build_player(pdict, loc, player_subs[pla], eve_str, decay)
        # if player is new, set just the player substitution and return
        if(pla not in pdict or pdict[pla].num_last7[loc+1] < 1):
            return pla_b
    
    # initialize the player values
    pla_a = build_player(pdict, loc, pla, eve_str, decay)
    # if player has played recently or doing pr evals
    if(pdict[pla].num_last7[loc+1] >= 2 or ispr):
        return pla_a

    # for players with only one recent event, use average of that event and player substitution
    for variable in vars(pla_a):
        setattr(pla_a, variable, (vars(pla_a)[variable] + vars(pla_b)[variable]) / 2)
    return pla_a

# load all data
def fill_pdict(nc=False, event=None):
    pdict = {}
    mypath = os.path.dirname(__file__)

    # select uncounted or counted events
    if(nc):
        txtfile = os.path.join(mypath,"stats_uncounted.txt")
    else:
        txtfile = os.path.join(mypath,"stats.txt")
    f = open(txtfile, "r")
    text = f.read()
    f.close()

    # if counted, update substitution stat files that use C2 stats
    if(not nc):
        mypath = mypath[:-2] + "Exterior Stats"
        txtfile = os.path.join(mypath, "C2stats.txt")
        f = open(txtfile, 'w')
        f.write(text)
        f.close()

    # read in all stats
    events = text.split("\n===\n")
    for e_num, eve_stats in enumerate(events):
        lines = eve_stats.split("\n")
        # for nc events, only read in the event with the relevant header
        if(nc):
            if(lines[0] != event):
                continue
            lines = lines[1:]
            # set e_num back to 0, only filling one entry for each stat
            e_num = 0

        total_stat2 = 0
        stat2s = []
        plas = []
        for j in range(len(lines)):
            words = lines[j].split("\t")
            name = words[0]
                        
            if(name not in pdict):
                # if not counted, initialize player with 1 entry per stat, else number of events
                pdict[name] = Player(1 if nc else len(events))
       
            # append all stats for the player in the event
            pdict[name].stat1[e_num] = float(words[1])
            pdict[name].stat2[e_num] = float(words[2])
            total_stat2 += float(words[2])
            pdict[name].stat3[e_num] = float(words[3])
            pdict[name].tstat1[e_num] = float(words[4])
            pdict[name].tstat2[e_num] = float(words[5])
            pdict[name].stat4[e_num] = (float(words[1])+0.3) / (float(words[4])+1.2)
            pdict[name].stat5[e_num] = float(words[2]) / max(float(words[5]), 0.01)
            stat2s.append(float(words[2]))
            plas.append(name)

            # logic for determining outlier "stat6" teams, checks results in stat2 for all 4 players
            if(j % 4 == 3):
                if(sorted(stat2s)[2] < 4):
                    for k in range(4):
                        pdict[plas[k]].stat6[e_num] = 1
                else:
                    for k in range(4):
                        pdict[plas[k]].stat6[e_num] = 0
                plas = []
                stat2s = []

        for pla in pdict:
            # if player in event, normalize stat2 to 40
            if(pdict[pla].stat2[e_num] != -1):
                pdict[pla].stat2[e_num] /= total_stat2/40
    
    if(nc):
        return pdict

    for pla in pdict:
        # determine number of previous events played in the last 7 for every event (plus current status)
        pdict[pla].num_last7.append(-1)
        for i in range(1, len(pdict[pla].num_last7)):
            eve_tot = 0
            # check last 7 events, or until start of event history
            for k in range(max(0, i-7), i):
                    if(pdict[pla].stat1[k] != -1):
                        eve_tot += 1
            # for early events, manually set to 2 if played in 1
            if(i < 3 and eve_tot == 1):
                eve_tot = 2
            pdict[pla].num_last7[i] = eve_tot

    return pdict

# iterates though events, finding average "strength" metric going into each event
# effectively an event comp measure 
def compute_eve_str(pdict, decay=default_decay):
    eve_str = [1] * 100
    new_eve_str = []
    for i in range(len(pdict["P053"].stat1)):
        new_eve_str.append(0)
        # only compute strength for events 9 and higher
        if(i >= 3):
            roster = all_rosters[i-3]
            # sum place average going into the event for all players
            for team in roster:
                for pla in team:
                    eve_pla = compute_eve_pla(pdict, i-1, pla, eve_str, decay, False)
                    new_eve_str[-1] += eve_pla.stat1
        else:
            new_eve_str[-1] = 1
    
    # find average over all events for normalization
    aver = np.mean(new_eve_str[3:])

    # normalize and scale all strengths
    for i in range(3, len(new_eve_str)):
        new_eve_str[i] = round(new_eve_str[i] / aver, 4)

    # set final (current) event to 0, neutral strength
    new_eve_str.append(1)

    return new_eve_str

def train(pdict, eve_str, eve_num=None, decay=default_decay, d=default_d):
    loc = 2
    matrix = []

    if(eve_num is None):
        eve_num = len(all_rosters) - 3

    temp_results = np.copy(results)

    # initialize stat priors for every team in every event
    for event in all_rosters[:3+eve_num]:
        average = 0
        # reduce impact on training for outlier events
        div = 3 if loc in reduced_events else 1
        for team in event:
            row = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            # for each player on team, add their stats to the team's stat row
            for pla in team:
                z = compute_eve_pla(pdict, loc, pla, eve_str, decay, False)
                row = np.add(row, [z.stat1, z.stat12, z.stat2, z.stat22, z.tstat1, z.tstat12, z.tstat2, z.tstat22, 
                    z.stat3, z.stat32, z.stat4, z.stat42, z.stat5, z.stat52, 1]) 
                average += z.stat2
            # append team result to training matrix
            matrix.append(list(np.divide(row,div*4)))
        average = average / 40
        # for all teams in event, append the "average performance metric" for the event 
        for j in range(10):
            matrix[-1 - j].append(average/div)
            # scale down training targets for reduced events as well
            if(div == 3):   
                temp_results[len(matrix)-1 - j] /= div
        
        loc += 1

    # select specific stats for training
    matrix = np.delete(matrix, d, axis=1)

    # solving for the weight vector
    tmatrix1 = np.transpose(matrix)
    fmatrix1 = np.matmul(np.linalg.inv(np.matmul(tmatrix1, matrix)), tmatrix1)
    weights = np.matmul(fmatrix1, temp_results[0:30+eve_num*10])

    return weights

def predict(pdict, eve_num, eve_str, decay=default_decay, d=default_d, roster=None):
    # get weights by training with all data prior to event
    weights = train(pdict, eve_str, eve_num, decay, d)

    # if roster is supplied, override event #eve_num with roster
    if(roster is not None):
        predict_roster = roster
    else:
        predict_roster = all_rosters[eve_num+3]
    
    input_matrix = []
    average = 0

    pdict_eve = {}
    # compute every player's stats going into the event
    for team in predict_roster:
        row = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        for pla in team:
            z = compute_eve_pla(pdict, eve_num+5, pla, eve_str, decay, False)
            pdict_eve[pla] = z

            row = np.add(row, [z.stat1, z.stat12, z.stat2, z.stat22, z.tstat1, z.tstat12, z.tstat2, z.tstat22, 
                z.stat3, z.stat32, z.stat4, z.stat42, z.stat5, z.stat52, 1]) 
            average += z.stat2
        input_matrix.append(list(np.divide(row,4)))

        
    average = average / 40
    # append average of the whole event as another trainable variable
    for j in range(10):
        input_matrix[j].append(average)
    
    # feature selection, delete stats numbered by d
    input_matrix = np.delete(input_matrix, d, axis=1)
    
    result = np.matmul(input_matrix, weights)

    norm = 10400 / sum(result)
    result = result * norm
    for j in range(10):
        result[j] = float(round(result[j]))
    
    # if no new roster, only return team results
    if(roster is None):
        return result, None

    indiv = []
    # Indiv scoring, 64% split evenly amongst the team, 36% individually based on stat2
    for j in range(10):
        tot = 0
        for pla in roster[j]:
            tot += pdict_eve[pla].stat2
        for pla in roster[j]:
            indiv.append(round(result[j]*0.16 + result[j]*0.36*(pdict_eve[pla].stat2/tot)))

    # return team and individual results
    return result, indiv

def generate_pr(pdict, eve_str, decay=default_decay, d=default_d):
    weights = train(pdict, eve_str, decay=decay, d=d)

    matrix_pr = []
    pr = []

    average = 0
    # compute stats for every player to identify their individual skill level
    for pla in pdict:
        z = compute_eve_pla(pdict, len(all_rosters)+2, pla, eve_str, decay, True)
        matrix_pr.append([z.stat1, z.stat12, z.stat2, z.stat22, z.tstat1, z.tstat12, z.tstat2, z.tstat22, 
                    z.stat3, z.stat32, z.stat4, z.stat42, z.stat5, z.stat52, 1]) 
        pr.append([pla])
        average += z.stat2
    average /= len(pdict)
    for i in range(len(matrix_pr)):
        matrix_pr[i].append(average)

    # using previously derived weight vector compute scores for every player
    matrix_pr = np.delete(matrix_pr, d, axis=1)
    scores = np.matmul(matrix_pr, weights)

    for i in range(len(pr)):
        pr[i].append(float(round(scores[i] / 4)))

    # sort player score pairs in descending order
    pr = (sorted(pr,key=lambda x: (x[1])))
    pr.reverse()

    # get the current list of players and their frequency playing
    f = open(os.path.join(os.path.split(os.path.dirname(__file__))[0], "current_list.txt"))
    text = f.read()
    lines = text.split("\n")
    pfreq_dict = {}
    for line in lines:
        values = line.split(" ")
        pfreq_dict[values[0]] = values[1]
    
    # remove players that aren't "current"
    ind = 0
    while(ind < len(pr)):
        if(pr[ind][0] not in pfreq_dict.keys()):
            del pr[ind]
        else:
            ind += 1

    avg_scores = 0
    total = 0
    for player in pr:
        # set all negative scores to 0
        if(player[1] < 0):
            player[1] = 0
        # compute frequency weighed average of the full player list
        avg_scores += player[1] * float(pfreq_dict[player[0]])
        total += float(pfreq_dict[player[0]])
    avg_scores /= total
    # scale all player scores to achieve a weighted average of 250
    for player in pr:
        player[1] = round(player[1] * (250 / avg_scores))
    
    return pr
    
if __name__ == "__main__":

    cla = C2_class()
    pdict = cla.pdict

    # all events leaderboard code
    perfList = []
    for key in cla.e_dict:
        if(key != "CUR"):
            val = cla.eve_pr(key)
            for value in val:
                perfList.append([value[1], value[0] + key[-2:]])
    perfList = sorted(perfList, reverse=True)

    # eve_pr test
    val = cla.eve_pr("EVE_18")

    # if current event, return pr
    if(cla.e_dict["CUR"] == eve_num):
        pr = generate_pr(cla.pdict, cla.eve_str, default_decay, default_d)
        string = ""
        for player in pr:
            string = string + (player[0] + "\t" + str(player[1]) + "\n")
        
        check = 0
    else:
        scores, indiv = predict(cla.pdict, eve_num, cla.eve_str, default_decay, default_d)

        # for simulated events, check against true results for mse accuracy
        results_test = results[30+eve_num*10:40+eve_num*10]
        error = 0
        errorm = []
        for j in range(10):
            error += np.power((scores[j] - results_test[j]), 2)
            errorm.append(scores[j] - results_test[j])
        error = error / 10
        error = np.sqrt(error)

        # evaluate mse accuracy of another prediction method for comparison
        error2 = 0
        error2m = []
        pr_preds_range = pr_preds[eve_num*10:10+eve_num*10]
        for j in range(10):
            error2 += np.power((pr_preds_range[j] - results_test[j]), 2)
            error2m.append(pr_preds_range[j] - results_test[j])
        error2 = error2 / 10
        error2 = np.sqrt(error2)

        # evaluate mse accuracy of naive prediction method for comparison
        error3 = 0
        norm_values = [1040, 1040, 1040, 1040, 1040, 1040, 1040, 1040, 1040, 1040]
        for j in range(10):
            error3 += np.power((norm_values[j] - results_test[j]), 2)
        error3 = error3 / 10
        error3 = np.sqrt(error3)

        errom = [int(x) for x in errorm]
        errom2 = [pow(x,2) for x in errom]
        error2m2 = [pow(x,2) for x in error2m]

        print(error)
        check = 1