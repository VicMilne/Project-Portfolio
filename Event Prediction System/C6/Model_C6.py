import numpy as np
import os.path

default_decay = 0.85
default_d = [1, 7, 9, 10, 12, 13, 14, 16]
eve_num = 21

event_select = [13,14,15,16,19,20,21,23,24,25,27,29,31,32,33,34,35,36,38,39,41,42,43]

results = [750, 1405, 1265, 690, 420, 1365, 385, 2215, 1295, 550, 940, 510, 1930, 480, 2130, 805, 850, 1135, 1005, 415,
    455, 955, 1115, 810, 1050, 1060, 1225, 1570, 1105, 780, 940, 1060, 815, 1270, 1230, 1520, 700, 400, 920, 925,
    1885, 1040, 960, 1160, 470, 1365, 415, 1630, 580, 645, 1255, 1135, 1290, 1415, 1855, 360, 970, 1110, 245, 410,
    470, 1570, 885, 715, 1175, 1085, 350, 805, 1755, 1400, 730, 1720, 465, 975, 630, 1255, 1080, 680, 815, 1695,
    1170, 1485, 765, 995, 470, 1275, 835, 910, 645, 1255, 790, 725, 1235, 1235, 1390, 515, 305, 390, 1575, 1700,
    965, 480, 830, 490, 715, 1220, 1480, 290, 1405, 2210, 835, 275, 855, 575, 2030, 1400, 1260, 650, 685, 1035,
    1180, 1255, 830, 450, 940, 1625, 510, 525, 1480, 1190, 1620, 1770, 1240, 780, 1050, 555, 540, 810, 705, 955,
    760, 1045, 1360, 840, 1020, 585, 635, 1075, 1645, 1290, 650, 1285, 680, 525, 1455, 1285, 1010, 905, 1200, 1085,
    545, 1025, 705, 760, 1665, 1275, 555, 1260, 1495, 545, 550, 930, 1175, 1405, 1185, 950, 1625, 1005, 775, 530,
    850, 950, 935, 1260, 1310, 1875, 500, 1175, 390, 970, 475, 1425, 1435, 1485, 830, 795, 975, 770, 1230, 430,
    1655, 360, 825, 930, 1420, 1030, 585, 1300, 1045, 795, 1815, 825, 945, 620, 1110, 525, 1075, 875, 750, 1165,
    450, 1415, 920, 985, 500, 1345, 1270, 1125, 1140, 685]

pr_preds = [
    980, 675, 1156, 1317, 1338, 951, 1084, 1583, 200, 933, 931, 1265, 718, 1088, 976, 1658, 790, 301, 1799, 663,
    1499, 556, 1429, 592, 1024, 374, 1065, 1105, 1561, 984, 848, 709, 623, 1654, 881, 581, 1495, 717, 1674, 980,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1761, 705, 1073, 1560, 254, 754, 1126, 975, 859, 1093,
    805, 1459, 944, 534, 1429, 1693, 550, 673, 1297, 701, 1088, 2007, 725, 801, 1225, 563, 630, 1196, 758, 1090,
    931, 1861, 863, 777, 1293, 780, 876, 587, 604, 1427, 819, 654, 833, 1190, 1721, 1117, 709, 1047, 200, 1709,
    722, 482, 953, 1101, 864, 1113, 754, 1046, 765, 2200, 1242, 973, 441, 980, 798, 881, 1595, 1916, 703, 469,
    832, 487, 952, 1657, 1109, 660, 867, 610, 1972, 855, 822, 413, 1334, 1113, 1443, 1613, 1396, 607, 223, 1037,
    1222, 674, 1920, 889, 931, 1255, 1057, 842, 733, 503, 729, 815, 635, 1205, 1739, 1575, 975, 515, 1023, 813,
    874, 1203, 931, 1493, 1662, 1352, 549, 663, 654, 643, 1094, 1752, 862, 725, 1561, 826, 691, 536, 1218, 759,
    1040, 1040, 1040, 1040, 1040, 1040, 1040, 1040, 1040, 1040, 583, 643, 1002, 636, 1371, 961, 498, 1540, 1379, 1412,
    1040, 1040, 1040, 1040, 1040, 1040, 1040, 1040, 1040, 1040]

player_subs = {"P046": "P034", "P010": "P019", "P029": "P074", "P062": "P061",
"P033": "P056", "P006": "P058", "P052": "P058", "P036": "P045", "P000": "P048", "P059": "P044",
"P060": "P055", "P023": "P026", "P028": "P029", "P005": "P035", "P022": "P049",
"P031": "P074", "P047": "P053", "P038": "P008", "P144": "P044", "P044": "P011", 
"P032": "P047", "P002": "P051", "P043": "P011", "P064": "P058", "P063": "P044", "P001": "P074",
"P035": "P024", "P067": "P035", "P016": "P000", "P004": "P051", "P025": "P054", "P125": "P067",
"P135": "P056", "P030": "P026", "P009": "P019", "P055": "P060", "P096": "P051", "P097": "P010",
"P099": "P010", "P133": "P021", "P124": "P055", "P020": "P042", "P050": "P062", "P137": "P049", "P039": "P029",
"P073": "P026", "P065": "P049", "P015": "P035", "P136": "P136", "P040": "P049", "P041": "P064",
"P027": "P005", "P054": "P048", "P075": "P053", "P086": "P059", "P134": "P049", "P120": "P004", 
"P121": "P032", "P037": "P039", "P115": "P038", "P152": "P015", "P148": "P060", "P143": "P038",
"P141": "P017", "P145": "P021", "P153": "P015", "P071": "P030", "P150": "P047", "P154": "P024",
"P147": "P032", "P155": "P051", "P151": "P010", "P149": "P033", "P146": "P006", "P105": "P141",
"P129": "P012", "P150": "P002", "P007": "P038", "P139": "P015", "P079": "P024", "P003": "P073",
"P156": "P009", "P078": "P026", "P140": "P073", "P138": "P033", "P077": "P030", "P076": "P030",
"P142": "P042", "P069": "P017", "P068": "P046", "P200": "P055", "P098": "P009", 
"P094": "P026", "P118": "P012", "P197": "P071", "P199": "P054", "P198": "P011", "P202": "P015",
"P203": "P051", "P179": "P118", "P204": "P009", "P178": "P053", "P205": "P071", "P188": "P057",
"P207": "P051", "P208": "P141", "P209": "P015", "P210": "P012", "P190": "P044", "P193": "P050", 
"P194": "P062", "P191": "P056", "P206": "P032", "P196": "P060", "P211": "P009", "P189": "P027", "P180": "P034",
"P181": "P000", "P212": "P073", "P213": "P141", "P195": "P042", "P214": "P032", "P223": "P011", 
"P224": "P055", "P225": "P141", "P105": "P051", "P226": "P038", "P227": "P064", "P220": "P054", 
"P219": "P054", "P230": "P056", "P228": "P012", "P229": "P136", "P130": "P063", "P021": "P062",
"P231": "P032", "P177": "P038", "P014": "P006", "P061": "P044", "P056": "P015", "P233": "P044",
"P079": "P026", "P186": "P015", "P072": "P060", "P208": "P073", "P232": "P202",
"P012": "P060"}

class C6_class:
    def __init__(self):
        self.pdict = fill_pdict()
        self.e_dict = {"EVE_11": -4, "EVE_13": -3, "EVE_14": -2, "EVE_15": -1, "EVE_16": 0, "EVE_18": 1, 
                       "EVE_21": 2, "EVE_22": 3, "EVE_23": 4, "EVE_25": 5, "EVE_26": 6, "EVE_27": 7, "EVE_29": 8,
                       "EVE_31": 9, "EVE_33": 10, "EVE_34": 11, "EVE_35": 12, "EVE_36": 13, "EVE_37": 14, 
                       "EVE_38": 15, "EVE_40": 16, "EVE_41": 17, "EVE_43": 18, "EVE_44": 19, "EVE_45": 20,
                       "CUR": 21}
        self.nc_eves = {"EVE_17": 6}
    # returns the predicted result of a previous event, with option to override with new roster
    def prev_sim(self, event, roster=None):
        results, _ = predict(self.pdict, self.e_dict[event], roster=roster)
        return results
    # returns player ranking results
    def pr_results(self):
        pr = generate_pr(self.pdict)
        return pr
    # returns predicted result of a new event, occuring at current time
    def cur_sim(self, roster):
        _, results = predict(self.pdict, self.e_dict["CUR"], roster=roster)
        return results
    # computes and returns single event player scores
    def eve_pr(self, event):
        if(event in self.nc_eves):
            # need the uncounted data for the above events
            dict_nc = fill_pdict(True, event)
            weights = train(self.pdict, self.nc_eves[event], d=[1, 7, 9, 10, 12, 13, 14, 16])
            return eve_pr_calc(dict_nc, -4, weights, d=[1, 7, 9, 10, 12, 13, 14, 16])
        
        ind = min(max(7, self.e_dict[event]+2), self.e_dict["CUR"])
        weights = train(self.pdict, d=[1, 7, 9, 10, 12, 13, 14, 16])
        return eve_pr_calc(self.pdict, self.e_dict[event], weights, [1, 7, 9, 10, 12, 13, 14, 16])

#############################################################################################

# instantiate all_rosters by reading from file
mypath = os.path.dirname(__file__)
txtfile = os.path.join(mypath[:-2], "All_Teamsets.txt")
f = open(txtfile, 'r')
lines = f.read().split("\n")
teams = [x.split("|") for x in lines]
all_rosters = [[x.split(", ") for x in y] for y in teams]

all_rosters = [all_rosters[i] for i in event_select]

class Player:
    def __init__(self, num_eve):
        self.stat1 = [-1] * num_eve
        self.stat5 = [-1] * num_eve
        self.stat3 = [-1] * num_eve
        self.tstat1 = [-1] * num_eve
        self.stat4 = [-1] * num_eve
        self.stat6 = [-1] * num_eve
        self.tstat6 = [-1] * num_eve
        self.comp = [-1] * num_eve
        self.num_teams = [-1] * num_eve
        self.num_last7 = [-1] * num_eve
        self.stat2 = [-1] * num_eve
        self.stat7 = [-1] * num_eve

class fullPlayer(Player):
    def __init__(self):
        self.stat1 = 0
        self.stat12 = 0
        self.stat3 = 0
        self.stat32 = 0
        self.tstat1 = 0
        self.tstat12 = 0
        self.stat4 = 0
        self.stat42 = 0
        self.stat5 = 0
        self.stat52 = 0
        self.stat6 = 0
        self.stat62 = 0
        self.tstat6 = 0
        self.tstat62 = 0
        self.stat8 = 0
        self.stat82 = 0
        self.stat9 = 0
        self.statA = 0
        self.stat7 = 0
        self.stat2 = 0


def eve_pr_calc(pdict, eve, weights, d=default_d):
    loc = eve + 4
    matrix = []
    plist = []

    # create stats matrix for every player in a particular event
    for pla in pdict:
        if(pdict[pla].stat1[loc] == -1):
            continue

        z = pdict[pla]
        matrix.append([z.stat1[loc]*2/3, pow(z.stat1[loc]*2/3, 2), z.stat3[loc], pow(z.stat3[loc], 2), z.tstat1[loc], pow(z.tstat1[loc], 2),  
                z.stat4[loc], pow(z.stat4[loc], 2), z.stat5[loc], pow(z.stat5[loc], 2), z.stat6[loc], z.tstat6[loc],
                (z.stat3[loc]/z.stat5[loc]), pow(z.stat3[loc]/z.stat5[loc], 2), z.stat1[loc]/max(z.tstat1[loc], 1), z.stat3[loc]/max(z.stat4[loc], 1), 
                z.stat7[loc], z.stat2[loc], 1, 3])
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
    vals *= f
    for i in range(len(plist)):
        plist[i].append(float(round(vals[i])))

    # sort plist by score, descending
    plist = (sorted(plist,key=lambda x: (x[1])))
    plist.reverse()
    return plist

def build_player(pdict, loc, pla, decay):
    inc = loc
    depth = 0
    div = 0
    outlier1 = [[],[],[],[],[],[],[],[],[],[]]
    outlier2 = []
    event_player = fullPlayer()

    a = 0.2
    # regularization stats
    event_player.stat3 += 5 * a
    event_player.stat32 += 28 * pow(a, 2)
    event_player.stat9 += 0.25 * a
    event_player.statA += 0.25 * a

    stats_list = ["stat1","stat12","tstat1","tstat12","stat4","stat42","stat5","stat52","stat6","stat62","tstat6","tstat62","stat8","stat82"]
    consts = [2, 8, 20, 20, 10, 40, 0.25]
    for i, stat in enumerate(stats_list):
        setattr(event_player, stat, pow(consts[i//2]*a, i % 2 + 1))

    div += 1 * a

    event_player.stat7 = 0 if pdict[pla].stat7[inc] == -1 else pdict[pla].stat7[inc]
    event_player.stat2 = 0 if pdict[pla].stat2[inc] == -1 else pdict[pla].stat2[inc]

    # move back in time, compute time decayed average of every stat
    while(inc >= 0):
        # comp level describes skill of each team faced, precomputed
        # sqrt to reduce effect
        comp_level = pow(abs(pdict[pla].comp[inc]), 1/2)

        # if stats for event don't exists (== -1) skip
        value = pdict[pla].stat1[inc]
        if value >= 0:

            # add 0.5 to normalize stat1
            value = (value + 0.5) * comp_level

            event_player.stat1 += value * pow(decay, depth)
            event_player.stat12 += pow(value,2) * pow(decay, depth)
            div += pow(decay,depth)
            if(depth < 5):
                outlier1[0].append(value * pow(decay, depth))
                # outlier2 only needs to be updated once, applies to all stats
                outlier2.append(pow(decay, depth))

            value = pdict[pla].stat3[inc] * comp_level
            event_player.stat3 += value * pow(decay, depth)
            event_player.stat32 += pow(value,2) * pow(decay, depth)
            if(depth < 5):
                outlier1[1].append(value * pow(decay, depth))

            value = pdict[pla].tstat1[inc] * comp_level
            event_player.tstat1 += value * pow(decay, depth)
            event_player.tstat12 += pow(value,2) * pow(decay, depth)
            if(depth < 5):
                outlier1[2].append(value * pow(decay, depth))

            value = pdict[pla].stat4[inc] * comp_level
            event_player.stat4 += value * pow(decay, depth)
            event_player.stat42 += pow(value,2) * pow(decay, depth)
            if(depth < 5):
                outlier1[3].append(value * pow(decay, depth))

            value = pdict[pla].stat5[inc] * comp_level
            event_player.stat5 += value * pow(decay, depth)
            event_player.stat52 += pow(value,2) * pow(decay, depth)
            if(depth < 5):
                outlier1[4].append(value * pow(decay, depth))

            value = pdict[pla].stat6[inc] * comp_level
            event_player.stat6 += value * pow(decay, depth)
            event_player.stat62 += pow(value,2) * pow(decay, depth)
            if(depth < 5):
                outlier1[5].append(value * pow(decay, depth))
            
            value = pdict[pla].tstat6[inc] * comp_level
            event_player.tstat6 += value * pow(decay, depth)
            event_player.tstat62 += pow(value,2) * pow(decay, depth)
            if(depth < 5):
                outlier1[6].append(value * pow(decay, depth))
            
            value = pdict[pla].stat3[inc] / pdict[pla].stat5[inc]
            event_player.stat8 += value * pow(decay, depth)
            event_player.stat82 += pow(value,2) * pow(decay, depth)
            if(depth < 5):
                outlier1[7].append(value * pow(decay, depth))

            value = pdict[pla].stat1[inc] / max(pdict[pla].tstat1[inc], 1)
            event_player.stat9 += value * pow(decay, depth)
            if(depth < 5):
                outlier1[8].append(value * pow(decay, depth))

            value = pdict[pla].stat3[inc] / max(pdict[pla].stat4[inc], 1)
            event_player.statA += value * pow(decay, depth)
            if(depth < 5):
                outlier1[9].append(value * pow(decay, depth))
       
        inc -= 1
        depth += 1
        
    # do outlier analysis if more than 3 events
    if(len(outlier2) > 3):
        # recompute original values using decayed value and amount of decay
        full_values = [outlier1[1][i] / outlier2[i] for i in range(len(outlier2))]
        mean = np.mean(full_values)
        std = np.std(full_values)

        for k in range(len(full_values)):
            # if stat2 is outlier 
            if(abs((full_values[k] - mean)/std) > 5):
                # iterate through variables in order defined in class, omitting substitution ones
                for i, var in enumerate(vars(event_player)):
                    # reduce outlier's effect by 50%, for squared values need to compute square first
                    if(i < 17):
                        if(i % 2 == 0):
                            setattr(event_player, var, vars(event_player)[var] - outlier1[i//2][k] * 0.5)
                        else:
                            setattr(event_player, var, vars(event_player)[var] - pow(outlier1[i//2][k], 2) * 0.5 / outlier2[k])
                    elif(i == 17):
                        setattr(event_player, var, vars(event_player)[var] - outlier1[8][k] * 0.5)
                div -= outlier2[k] * 0.5

    # iterate through all variables, divide by sum of weights
    for i, var in enumerate(vars(event_player)):
        if(i > 17):
            break
        setattr(event_player, var, vars(event_player)[var] / div)

    return event_player

def compute_eve_pla(pdict, loc, pla, decay, ispr):
    # if player not initialized or (haven't played many recent events and doing predictions)
    if(pla not in pdict or (pdict[pla].num_last7[loc+1] < 2 and not ispr)):
        # determine player substitution values
        pla_b = build_player(pdict, loc, player_subs[pla], decay)
        # if player is new, set just the player substitution and return
        if(pla not in pdict or pdict[pla].num_last7[loc+1] < 1):
            return pla_b
    
    # initialize the player values
    pla_a = build_player(pdict, loc, pla, decay)
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

    # read in all stats
    events = text.split("\n===\n")
    for enum, eve_stats in enumerate(events):
        lines = eve_stats.split("\n")
        # for nc events, only read in the event with the relevant header
        if(nc):
            if(lines[0] != event):
                continue
            lines = lines[1:]
            # only reading on one set of values, into first position in arrays
            enum = 0

        incr = 0
        team_pla = []
        tot_stat1 = 0
        tot_stat3 = 0
        tot_m8at6 = 0
        for line in lines:
            words = line.split("\t")
            name = words[0]
            if(name not in pdict):
                # if not counted, initialize player with 1 entry per stat, else number of events
                pdict[name] = Player(1 if nc else len(events))
      
            pdict[name].stat1[enum] = int(words[1])
            pdict[name].stat5[enum] = float(words[3])
            pdict[name].stat3[enum] = int(words[2])
            # maintain running tally of stat1 and .stat3 for each team   
            tot_stat1 += int(words[1])
            tot_stat3 += int(words[2])
            pdict[name].stat6[enum] = int(words[5])
            tot_m8at6 += int(words[5])
            pdict[name].num_teams[enum] = len(words[6])//2+1
            pdict[name].comp[enum] = float(words[7])
            team_pla.append(name)
            if(incr % 4 == 3):
                # once all 4 members of team processed, add totals to every player
                for pla in team_pla:
                    pdict[pla].tstat1[enum] = tot_stat1
                    pdict[pla].stat4[enum] = tot_stat3
                    pdict[pla].tstat6[enum] = tot_m8at6 * pdict[pla].stat3[enum] / tot_stat3
                # reset totals for next team
                team_pla = []
                tot_stat1 = 0
                tot_stat3 = 0
                tot_m8at6 = 0
            incr += 1

    if(nc):
        return pdict

    # files and variables for substitution data from each contest    
    file_list = ["C6_C9", "C6_C2"]
    var_list = ["stat2", "stat7"]

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
            # read only the 2nd and later events, first 6 are not trained on
            for j in range(len(vals) - 1):
                getattr(pdict[pla], var_list[i])[j] = float(vals[j+1])
        f.close()
    
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
            if(i < 2 and eve_tot == 1):
                eve_tot = 2
            pdict[pla].num_last7[i] = eve_tot

    return pdict

def train(pdict, eve_num=None, decay=default_decay, d=default_d):
    matrix = []

    if(eve_num is None):
        eve_num = len(all_rosters) - 2
    
    loc = max(0,eve_num-10) + 1

    # initialize stat priors for every team in every event  
    for event in all_rosters[max(0,eve_num-10):2+eve_num]:
        average = 0
        for team in event:
            row = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            # for each player on team, add their stats to the team's stat row
            for pla in team:
                z = compute_eve_pla(pdict, loc, pla, decay, False)
                row = np.add(row, [z.stat1, z.stat12, z.stat3, z.stat32, z.tstat1, z.tstat12, z.stat4, 
                    z.stat42, z.stat5, z.stat52, z.stat6, z.tstat6, z.stat8, z.stat82, z.stat9,
                      z.statA, z.stat7, z.stat2, 1])
                average += z.stat1
            # append team result to training matrix
            row = np.divide(row, 4)
            matrix.append(list(row))

        average = average / 40
        # for all teams in event, append the "average performance metric" for the event 
        for j in range(10):
            matrix[-1 - j].append(average)
        loc += 1

    # select specific stats for training
    matrix = np.delete(matrix, d, axis=1)

    # perform linear regression with regularization to prevent overfitting
    weights = [1000000]
    reg = 0
    while(max(weights) > 1000 or min(weights) < -1000):
        ident = np.identity(len(matrix[0])) * reg
        tmatrix0 = np.transpose(matrix)
        fmatrix0 = np.matmul(np.linalg.inv(np.matmul(tmatrix0, matrix) + ident), tmatrix0)
        weights = np.matmul(fmatrix0, results[max(0,eve_num-10)*10:20+eve_num*10])
        reg += 0.05

    # return weight vector
    return weights

def predict(pdict, eve_num, decay=default_decay, d=default_d, roster=None):
    # get weights by training with all data prior to event
    weights = train(pdict, eve_num, decay, d)

    # if roster is supplied, override event #eve_num with roster
    if(roster is not None):
        predict_roster = roster
    else:
        predict_roster = all_rosters[eve_num+2]
    
    input_matrix = []
    average = 0

    pdict_eve = {}
    # compute every player's stats going into the event
    for team in predict_roster:
        row = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        for pla in team:
            z = compute_eve_pla(pdict, eve_num+3, pla, decay, False)
            pdict_eve[pla] = z

            row = np.add(row, [z.stat1, z.stat12, z.stat3, z.stat32, z.tstat1, z.tstat12, z.stat4, 
                    z.stat42, z.stat5, z.stat52, z.stat6, z.tstat6, z.stat8, z.stat82, z.stat9,
                      z.statA, z.stat7, z.stat2, 1]) 
            average += z.stat1
        row = np.divide(row, 4)
        input_matrix.append(list(row))
    
    average = average / 40
    # append average time of the whole event as another trainable variable
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
    # Indiv scoring, 76% split evenly amongst the team, 24% individually based on stat2
    for j in range(10):
        tot = 0
        for pla in roster[j]:
            tot += pdict_eve[pla].stat3
        for pla in roster[j]:
            indiv.append(round(result[j]*0.19 + result[j]*0.24*(pdict_eve[pla].stat3/tot)))
    
    return result, indiv
    
def generate_pr(pdict, decay=default_decay, d=default_d):
    # get weights by training with all events
    weights = train(pdict, decay=decay, d=d)

    matrix_pr = []
    pr = []

    pla_dict = {}
    average = 0
    # compute stats for every player to identify their individual skill level
    for pla in pdict:
        z = compute_eve_pla(pdict, len(all_rosters)+1, pla, decay, True)
        pla_dict[pla] = z
        matrix_pr.append([z.stat1, z.stat12, z.stat3, z.stat32, z.tstat1, z.tstat12, z.stat4, 
                    z.stat42, z.stat5, z.stat52, z.stat6, z.tstat6, z.stat8, z.stat82, z.stat9,
                      z.statA, z.stat7, z.stat2, 1])
        average += z.stat1
        pr.append([pla])

    average /= len(pdict)
    for row in matrix_pr:
        row.append(average)
    
    # using previously derived weight vector compute scores for every player
    matrix_pr = np.delete(matrix_pr, d, axis=1)
    scores = np.matmul(matrix_pr, weights)

    for i in range(len(pr)):
        pr[i].append(float(round(scores[i])))

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

    cla = C6_class()

    # all events leaderboard code
    perfList = []
    for key in cla.e_dict:
        if(key != "CUR"):
            val = cla.eve_pr(key)
            for value in val:
                perfList.append([value[1], value[0] + key[-2:]])
    perfList = sorted(perfList, reverse=True)

    val = cla.eve_pr("EVE_45")

    # if current event, return player rankings
    if(cla.e_dict["CUR"] == eve_num):
        pr = generate_pr(cla.pdict, default_decay, default_d)
        string = ""
        for player in pr:
            string = string + (player[0] + "\t" + str(player[1]) + "\n")
        
        check = 0
    else:
        scores, indiv = predict(cla.pdict, eve_num, default_decay, default_d)

        # for simulated events, check against true results for mse accuracy
        results_test = results[20+eve_num*10:30+eve_num*10]
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


    # FEATURE SELECTION CODE
    
    # matrix = np.delete(matrix, full_d, axis=1)
    
    # from itertools import chain, combinations

    # def powerset(iterable):
    #     "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    #     s = list(iterable)
    #     return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    # y = [0,1,2,3,4,5,6,7,8,9,10,11,12]
    # c = powerset(y)
    # d = list(c)
    # del d[len(d) - 1]
    # overall_error = []
    # for m in range(1, 6):
    #     for i in range(377):
    #         temp_matrix = np.array(np.copy(matrix))
    #         temp_matrix = np.delete(temp_matrix, list(d[i]), axis=1)
    #         matrix1 = temp_matrix[:(m*-10)]
    #         # perform linear regression with regularization to prevent overfitting
    #         weights = [1000000]
    #         reg = 0
    #         while(max(weights) > 500 or min(weights) < -500):
    #             ident = np.identity(len(matrix1[0])) * reg
    #             tmatrix0 = np.transpose(matrix1)
    #             fmatrix0 = np.matmul(np.linalg.inv(np.matmul(tmatrix0, matrix1) + ident), tmatrix0)
    #             weights = np.matmul(fmatrix0, results[:20+eve_num*10-(m*10)])
    #             reg += 0.05

    #         test = np.matmul(temp_matrix[20+eve_num*10-(m*10):30+eve_num*10-(m*10)], weights)
    #         norm = 10000 / sum(test)
    #         test = test * norm
    #         results_test = results[20+eve_num*10-(m*10):30+eve_num*10-(m*10)]
    #         error = 0
    #         for j in range(10):
    #             test[j] = float(round(test[j]))
    #             error += np.power((test[j] - results_test[j]), 2)
    #         error = error / 10
    #         error = np.sqrt(error)
    #         print(error)
    #         tot_err = error 
    #         if(m == 1):
    #             overall_error.append([tot_err, d[i]])
    #         else:
    #             overall_error[i][0] += tot_err
    
    # overall_error = sorted(overall_error, key=lambda x: x[0])