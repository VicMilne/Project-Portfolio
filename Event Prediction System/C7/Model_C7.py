import numpy as np
import os.path

decay = 0.85
default_d = [4,6,8,10,11]

use_subs = False
 
eve_num = 12

# numbered events where this contest appeared
event_select = [0, 2, 3, 6, 8, 9, 10, 12, 14, 17, 20, 24, 26, 29, 35, 37, 41]

thres = 0.1

results = [975,1050,868,1083,746,1448,685,249,1391,1401,1595,840,974,1188,974,1318,647,715,706,1094,
    258,1224,1430,566,762,929,1020,1083,1398,1177,654,723,1120,665,1151,1384,772,677,1176,1096,
    1121,1584,739,1140,1115,756,689,666,505,1500,1153,1478,1016,799,1018,556,1090,1274,757,763,
    1152,675,1319,1012,922,767,868,863,1063,920,1058,1047,802,789,733,876,845,643,1426,1395,
    1225,385,811,1156,1196,682,922,910,1111,1303,575,1033,980,941,1140,984,1400,1316,944,380,
    645,1374,1014,587,633,1277,727,974,1117,1073,859,848,682,1030,890,1773,1230,950,598,1100,
    968,564,1258,718,1379,1434,1021,970,1104,624,1125,1027,1130,1078,196,1327,1094,960,1150,767,
    836,296,465,1426,1373,1366,1184,1198,1238,557]

pr_preds = [980,980,980,980,980,980,980,980,980,980,980,980,980,980,980,980,980,980,980,980,
    1006,1520,1118,660,956,621,615,926,1534,816,1404,1333,1117,1182,820,600,1044,389,939,970,
    778,954,1323,898,1167,875,874,571,1492,773,1271,985,1075,1003,1074,913,1376,464,709,793,
    544,808,1282,953,1063,1218,941,570,992,1328,868,1174,1189,717,770,1419,893,717,980,959,
    1179,921,895,830,1000,1435,768,889,785,1378,1171,658,1274,799,1553,722,1118,800,843,807,
    885,1031,930,844,377,1455,860,1203,1234,927,980,980,980,980,980,980,980,980,980,980]

player_subs = {"P078": "P019", "P031": "P022", "P124": "P011", "P125": "P025", "P047": "P033", 
"P012": "P058", "P057": "P051", "P008": "P019", "P084": "P035", "P144": "P054", "P017": "P019", 
"P117": "P033", "P109": "P127", "P096": "P127", "P107": "P033", "P038": "P012", "P100": "P033", 
"P064": "P019", "P000": "P059", "P046": "P045", "P016": "P017", "P026": "P017", "P023": "P019", 
"P004": "P044", "P055": "P011", "P043": "P014", "P130": "P017", "P056": "P058", "P067": "P044", 
"P024": "P127", "P126": "P144", "P021": "P144", "P136": "P127", "P032": "P144", "P005": "P051", 
"P028": "P000", "P063": "P012", "P002": "P032", "P030": "P019", "P009": "P011", "P014": "P042", 
"P020": "P022", "P099": "P127", "P097": "P017", "P034": "P031", "P050": "P059", "P052": "P006", 
"P137": "P011", "P039": "P059", "P065": "P042", "P015": "P009", "P073": "P134", "P059": "P043", 
"P060": "P067", "P040": "P049", "P027": "P016", "P134": "P032", "P086": "P033", "P120": "P017", 
"P121": "P009", "P075": "P026", "P071": "P096", "P148": "P067", "P141": "P016", "P143": "P038", 
"P145": "P065", "P150": "P004", "P154": "P026", "P153": "P099", "P152": "P017", "P149": "P036", 
"P146": "P056", "P147": "P012", "P155": "P017", "P151": "P017", "P041": "P051", "P019": "P035", 
"P006": "P035", "P010": "P017", "P068": "P001", "P077": "P097", "P203": "P008", "P202": "P015",
"P204": "P016", "P178": "P144", "P179": "P147", "P205": "P024", "P210": "P060", "P213": "P039", 
"P216": "P026", "P217": "P056", "P218": "P097", "P215": "P097", "P220": "P144", "P221": "P005", 
"P219": "P049", "P222": "P062", "P098": "P002", "P214": "P049", "P224": "P025", "P225": "P136",
"P226": "P016", "P231": "P038", "P142": "P049", "P139": "P051", "P140": "P056", 
"P227": "P025", "P025": "P060", "P118": "P056"}

class C7_class:
    def __init__(self):
        self.pdict = fill_pdict()
        self.e_dict = {"EVE_1": -5, "EVE_3": -4, "EVE_4": -3, "EVE_7": -2, "EVE_9": -1, "EVE_10": 0,
                       "EVE_11": 1, "EVE_13": 2, "EVE_15": 3, "EVE_19": 4, "EVE_22": 5, "EVE_26": 6, 
                       "EVE_28": 7, "EVE_31": 8, "EVE_37": 9, "EVE_39": 10, "EVE_43": 11, "CUR": 12}
        self.nc_eves = {"EVE_2": 4, "EVE_12": 4, "EVE_17": 6, "EVE_23": 8}
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
        results_indiv = []
        for val in results:
            for i in range(4): results_indiv.append(round(val/4))
        return results_indiv
    # computes and returns single event player scores
    def eve_pr(self, event):
        if(event in self.nc_eves):
            weights = train(self.pdict, eve_num=self.nc_eves[event], d=[6,7,8,9,10,11])
            # need the uncounted data for the above events
            dict_nc = fill_pdict(True, event)
            return eve_pr_calc(dict_nc, -5, weights, d=[6,7,8,9,10,11])
        
        ind = min(max(6, self.e_dict[event]+2), self.e_dict["CUR"])
        weights = train(self.pdict, eve_num=ind, d=[6,7,8,9,10,11])
        return eve_pr_calc(self.pdict, self.e_dict[event], weights, d = [6,7,8,9,10,11])

#############################################################################################
# instantiate all_rosters by reading from file
mypath = os.path.dirname(__file__)
txtfile = os.path.join(mypath[:-2], "All_Teamsets.txt")
f = open(txtfile, 'r')
lines = f.read().split("\n")
teams = [x.split("|") for x in lines]
all_rosters = [[x.split(", ") for x in y] for y in teams]

all_rosters = [all_rosters[i] for i in event_select]
all_rosters[1][1][2] = "P042"
all_rosters[3][1][3] = "P066"

nc_roster = [[["P054", "P086", "P079", "P010"], ["P059", "P032", "P041", "P157"],
    ["P159", "P158", "P161", "P160"], ["P061", "P062", "P066", "P082"],
    ["P162", "P163", "P164", "P165"], ["P049", "P126", "P166", "P167"],
    ["P031", "P047", "P168", "P115"], ["P053", "P051", "P127", "P169"],
    ["P106", "P120", "P100", "P017"], ["P104", "P025", "P057", "P084"]],
    [["P013", "P014", "P021", "P132"], ["P031", "P032", "P016", "P056"],
    ["P053", "P025", "P023", "P051"], ["P036", "P126", "P043", "P017"],
    ["P150", "P167", "P136", "P170"], ["P000", "P062", "P005", "P064"],
    ["P001", "P059", "P063", "P115"], ["P067", "P096", "P099", "P097"],
    ["P034", "P035", "P010", "P127"], ["P074", "P048", "P172", "P171"]],
    [["P028", "P041", "P079", "P024"], ["P002", "P064", "P129", "P138"],
    ["P033", "P063", "P109", "P173"], ["P168", "P115", "P082", "P083"],
    ["P021", "P150", "P035", "P174"], ["P175", "P037", "P176", "P156"],
    ["P048", "P005", "P133", "P170"], ["P004", "P043", "P016", "P127"],
    ["P020", "P008", "P030", "P197"], ["P162", "P163", "P164", "P165"]],
    [["P042", "P025", "P023", "P153"], ["P032", "P040", "P053", "P027"],
    ["P034", "P005", "P019", "P051"], ["P058", "P039", "P035", "P024"],
    ["P029", "P048", "P150", "P154"], ["P031", "P064", "P177", "P139"],
    ["P044", "P059", "P063", "P017"], ["P001", "P067", "P096", "P097"],
    ["P036", "P145", "P016", "P115"], ["P000", "P009", "P062", "P146"]]]

nc_nums = {"EVE_2": 0, "EVE_12": 1, "EVE_17": 2, "EVE_23": 3}

class Player:
    def __init__(self):
        self.num_stat1 = []
        self.stat4 = []
        self.stat11 = []
        self.stat12 = []
        self.stat13 = []
        self.stat2 = []
        self.stat3 = []
        self.stat5 = []
        self.place = []
        self.num_last7 = []
        self.stat3_unadj = []
        self.stat2_unadj = []
        self.stat1_unadj = []

class eventPlayer:
    def __init__(self):
        self.stat1 = 0
        self.stat12 = 0
        self.num_stat1 = 0
        self.stat2 = 0
        self.stat22 = 0
        self.stat3 = 0
        self.stat32 = 0
        self.stat5 = 0

# factors are 11/54, 12/54, 13/54, 14/54, 15/54, normalized version of 9/54, 11/54, 13/54, 15/54, 17/54
def eve_pr_calc(pdict, eve, ans, d=default_d):
    loc = eve + 5
    matrix = []
    plist = []

    # store additional attack and defense times for reference
    stat2_stand = []
    stat1_stand = []

    # create stats matrix of every player in the particular event
    for pla in pdict:
        z = pdict[pla]
        if(z.stat3[loc] == -1):
            continue
        nh = z.num_stat1[loc]

        # temporarily store stat results for debug purposes
        stat2_stand.append([pla, (z.stat3[loc] + z.stat2[loc]*2)/3])
        if(nh == 0):
            stat1_stand.append([pla, -1])
        else:
            stat1_stand.append([pla, 1/z.stat13[loc]])

        stat3 = pow(z.stat3[loc], 3/4)
        # lightly reduce importance of stat2 relative to other stats for pr purposes
        if(nh == 0):
            stat2 = pow(z.stat2[loc], 3/4)
        else:
            stat2 = pow(z.stat2[loc], 3/4) * (20-nh)/20 + pow(1/z.stat13[loc], 3/4) * (nh)/20

        # various weights for 0, 1, 2, 3, or 4 attacks
        if(nh == 0):
            x1, x2, x3, x4 = 1/3, 0, 11/54, 1
        elif(nh == 1):
            x1, x2, x3, x4 = 8/27, 1/9, 12/54, 8/11
        elif(nh == 2):
            x1, x2, x3, x4 = 7/27, 2/9, 13/54, 7/13
        elif(nh == 3):
            x1, x2, x3, x4 = 6/27, 3/9, 14/54, 6/15
        else:
            x1, x2, x3, x4 = 5/27, 4/9, 15/54, 5/17
        # initialize combined stats using implied "average" result (=1) for other half of equation
        comb_ind = stat3 * x4 + (1/z.stat13[loc]) * (1-x4)
        comb_tot = stat2 * x4 + (1/z.stat13[loc]) * (1-x4)

        matrix.append([stat3 * x1, pow(stat3, 2) * x1, stat2 * x1, pow(stat2, 2) * x1, z.stat5[loc] * x1,
                                     (1/z.stat13[loc]) * x2, pow((1/z.stat13[loc]), 2) * x2, comb_ind*x3, pow(comb_ind, 2)*x3,
                                      comb_tot*x3, pow(comb_tot, 2)*x3, nh, 0.25, 0.6])

        plist.append([pla])
    
    matrix = np.delete(matrix, d, axis=1)

    vals = np.matmul(matrix, ans)

    # formula for adjusting all event scores to be above 30, maintains relative positions
    x = min(vals)
    if(x < 60):
        m = 30 / (60 - x)
        b = 60 - 60*m
        for i in range(len(vals)):
            if(vals[i] < 60):
                vals[i] = vals[i]*m + b

    # normalize and round, then pair scores with players
    f = 10000 / sum(vals)
    vals *= f
    for i in range(len(plist)):
        plist[i].append(float(round(vals[i])))


    plist = (sorted(plist,key=lambda x: (x[1]), reverse=True))
    stat2_stand = sorted(stat2_stand, key=lambda x: (x[1]), reverse=True)
    stat1_stand = sorted(stat1_stand, key=lambda x: (x[1]), reverse=True)

    return plist

# compute average times for 3 stats in event "index", weighted by frequency each player attacks vs defenses
def calculate_average(pdict, index):
    aver1, aver2, aver3 = 0, 0, 0
    tot1, tot2, tot3 = 0, 0, 0
    for pla in pdict:
        if(pdict[pla].stat3[index] != -1):
            aver1 += pdict[pla].stat3_unadj[index] * pdict[pla].stat4[index]
            tot1 +=  pdict[pla].stat4[index]
        if(pdict[pla].stat13[index] != -1):
            aver2 += pdict[pla].stat1_unadj[index] * pdict[pla].num_stat1[index]
            tot2 += pdict[pla].num_stat1[index]
        if(pdict[pla].stat2[index] != -1):
            aver3 += pdict[pla].stat2_unadj[index] * pdict[pla].stat4[index]
            tot3 +=  pdict[pla].stat4[index]

    return [aver1/tot1, aver2/tot2, aver3/tot3]

def attacker_check(pdict, name, aver, enum, decay=decay):
    # if 1st event (no baseline performance) return default strength of 1 (average)
    if(len(aver) == 0):
        return 1
    
    # if player lacks past data (or data is limited) use player substitution
    if(max(pdict[name].stat13[:enum]) == -1 or sum([x if x != -1 else 0 for x in pdict[name].num_stat1[:enum]]) < 3):
        if(len(aver) <= 2):
            return 1
        return attacker_check(pdict, player_subs[name], aver, enum, decay=decay)
    
    # compute average attack performance in past events, weighted by # of rounds attacking, time decayed
    total = 0.6
    div = 0.5
    depth = 0

    times = pdict[name].stat1_unadj
    stat1s = pdict[name].num_stat1
    # iterate through all past times (not current event) in reverse
    for i in range(enum - 1, -1, -1):
        # if played in this event, add to running sum
        if(times[i] != -1):
            # normalized attack time, weighted by number of attack rounds and time decayed
            total += (times[i] / aver[i][1]) * stat1s[i] * pow(decay,depth)
            div += stat1s[i] * pow(decay,depth)
        depth += 1

    # compute weighted average, return inverse (threshold of 0.3)
    total /= div
    if(1/total < 0.3):
        return 0.3
    
    return pow(1/total, 3/4)

# iterate through all defenders (max=3) and compute average defense performance
def defenders_check(pdict, names, aver, enum, decay=decay):
    # if 1st event (no baseline performance) return default strength of 1 (average)
    if(len(aver) == 0):
        return 1
    
    total = 0

    # iterate through all players on team
    for i in range(len(names)):
        # if player lacks past data
        if(max(pdict[names[i]].stat3[:enum]) == -1):
            # for first two events just use average performance
            if(len(aver) < 2):
                total += 1
            else:
                # otherwise use substitution
                # must undo reciprocal and norm performed at end of check
                total += 1/pow(defenders_check(pdict, [player_subs[names[i]]], aver, enum, decay), 4/3)
            continue
        
        # running total for the player
        pla_tot = 0
        # sum of the weights
        div = 0

        depth = 0
        defends = pdict[names[i]].stat2_unadj
        # iterate through all past times (not current event) in reverse
        for j in range(enum-1, -1, -1):
            # if played in this event, add to running sum
            if(defends[j] != -1):
            # normalized defend time, time decayed
                pla_tot += (defends[j] / aver[j][2]) * pow(decay,depth)
                div += pow(decay,depth)
            depth += 1
        total += pla_tot / div

    total /= len(names)
    return pow(1/total, 3/4)

# computes time decayed averages for all stats up to time loc
# performs outlier analysis
def build_player(pdict, loc, pla, decay):
    inc = loc
    depth = 0
    div = [0,0,0]
    outlier1 = [[],[],[],[]]
    outlier2 = [[],[]]
    event_player = eventPlayer()

    for var in ["stat1", "stat12", "stat3", "stat32", "stat2", "stat22"]:
        setattr(event_player, var, 0.4)
    div[0] = 0.5
    div[1] = 0.5
    
    while(inc >= 0):

        ### attack stats
        value = 1 / pdict[pla].stat13[inc]
        if value != -1:
            event_player.stat1 += value * pow(decay, depth)
            event_player.stat12 += pow(value,2) * pow(decay, depth)
            div[0] += pow(decay,depth)
            if(depth < 5):
                outlier1[0].append(value * pow(decay, depth))
                outlier2[0].append(pow(decay, depth))

        ### defense stats
        value = pdict[pla].stat2[inc]
        if value != -1:
            event_player.stat2 += value * pow(decay, depth)
            event_player.stat22 += pow(value,2) * pow(decay, depth)
            div[1] += pow(decay,depth)
            if(depth < 5):
                outlier1[1].append(value * pow(decay, depth))
                outlier2[1].append(pow(decay, depth))
        
            value = pdict[pla].stat3[inc]
            event_player.stat3 += value * pow(decay, depth)
            event_player.stat32 += pow(value,2) * pow(decay, depth)
            if(depth < 5):
                outlier1[2].append(value * pow(decay, depth))

            value = pdict[pla].num_stat1[inc]
            event_player.num_stat1 += value * pow(decay, depth)
            div[2] += pow(decay,depth)
        
            value = pdict[pla].stat5[inc]
            event_player.stat5 += value * pow(decay, depth)
            if(depth < 5):
                outlier1[3].append(value * pow(decay, depth))
       
        inc -= 1
        depth += 1

    # do outlier analysis if more than 3 events
    if(len(outlier2) > 3):
        # first check and adjust for attacker outliers
        full_values = [outlier1[0][i] / outlier2[0][i] for i in range(len(outlier2[0]))]
        mean = np.mean(full_values)
        std = np.std(full_values)
        for k in range(len(full_values)):
            # reduce outlier's effect by 50%, for squared values need to compute first
            if(abs((full_values[k] - mean)/std) > 10):
                event_player.stat1 -= outlier1[0][k] * 0.5
                event_player.stat12 -= pow(outlier1[0][k], 2) * 0.5 / outlier2[k]
                div[0] -= outlier2[k] * 0.5
        
        # then address defending outlier stats
        full_values = [outlier1[1][i] / outlier2[1][i] for i in range(len(outlier2[1]))]
        mean = np.mean(full_values)
        std = np.std(full_values)
        for k in range(len(full_values)):
            # if num is outlier 
            if(abs((full_values[k] - mean)/std) > 10):
                # iterate through all variables, skipping first 3 (attacking)
                for i, var in enumerate(vars(event_player)):
                    # skip first 3 attack stats (i > 2)
                    if(i > 2):
                        # odd variables are standard, even are squared feature maps
                        if(i % 2 == 1):
                            setattr(event_player, var, vars(event_player)[var] - outlier1[(i-1)//2][k] * 0.5)
                        else:
                            setattr(event_player, var, vars(event_player)[var] - pow(outlier1[(i-1)//2][k], 2) * 0.5 / outlier2[k])
                div[1] -= outlier2[k] * 0.5

    # normalize attack and defend times seperately
    if(div[0] != 0):
        event_player.stat1 /= div[0]
        event_player.stat12 /= div[0]
    
    if(div[1] != 0):
        ev_stats = ["stat2", "stat22", "stat3", "stat32", "stat5"]
        for stat in ev_stats:
            setattr(event_player, stat, vars(event_player)[stat] / div[1])
        
    event_player.num_stat1 /= div[2]

    return event_player


def compute_eve_pla(pdict, loc, pla, decay, ispr):
    # if player not initialized or (haven't played many recent events and doing predictions)
    if(pla not in pdict or (pdict[pla].num_last7[loc] < 2 and not ispr)):
        # determine player substitution values
        if(use_subs or loc+1 == len(pdict[pla].stat2)):
            pla_b = build_player(pdict, loc, player_subs[pla], decay)
        else:
            # if not using subs, seed with data from the current event
            pla_b = build_player(pdict, loc+1, pla, decay)

        # if player is new, set just the player substitution and return
        if(pla not in pdict or pdict[pla].num_last7[loc] < 1):
            return pla_b
    
    # initialize the player values
    pla_a = build_player(pdict, loc, pla, decay)
    # if player has played recently or doing pr evals
    if(pdict[pla].num_last7[loc] >= 2 or ispr):
        return pla_a

    # for players with only one recent event, use average of that event and player substitution
    for variable in vars(pla_a):
        setattr(pla_a, variable, (vars(pla_a)[variable] + vars(pla_b)[variable]) / 2)
    return pla_a

def load_match(pdict, enum, baseline, attacker, defenders_list, decay=decay):
    # get attacker strength
    astrength = attacker_check(pdict, attacker, baseline, enum, decay=decay)

    prev_elim = 0
    place = 0
    for pla, time in defenders_list:
        # if a successful defense is found in timeline, break from loop
        if(defenders_list[place][1] == 60):
            break

        place += 1
        if(astrength > thres):
            # update current defender with their defend time
            pdict[pla].stat4[enum] += 1
            pdict[pla].place[enum] += place
            pdict[pla].stat2[enum] += time * astrength
            pdict[pla].stat2_unadj[enum] += time
            # compare with previous elim time to find stat3
            pdict[pla].stat3[enum] += (time-prev_elim) * astrength
            pdict[pla].stat3_unadj[enum] += (time-prev_elim)
            prev_elim = time
            # if last player elim, update all shared times with overall time
            if(place == 3):
                for pla, _ in defenders_list:
                    pdict[pla].stat5[enum] += time * astrength

        # update corresponding variable (stat11, stat12, or stat13) for the attacker
        var_name = "stat1" + str(place)
        defend_score = defenders_check(pdict, [x[0] for x in defenders_list[:place]], baseline, enum, decay)
        getattr(pdict[attacker], var_name)[enum] += time * defend_score
        if(place == 3):
            pdict[attacker].stat1_unadj[enum] += time

    # if all 3 defenders eliminated, no more work to be done
    if(place == 3):
        return

    ### deal with successful defenses and corresponding attacker times

    # precompute overall defender strength
    dstrength = defenders_check(pdict, [x[0] for x in defenders_list], baseline, enum, decay)

    # use preset defend and attack times for each possible amount of successful defenders
    if(place == 2):
        stat2, stat3, stat5 = 70, 70 - defenders_list[1][1], max(70, defenders_list[1][1]*3/2)
        pdict[attacker].stat13[enum] += max(70, defenders_list[1][1]*3/2) * dstrength
        pdict[attacker].stat1_unadj[enum] += stat5
    elif(place == 1):
        stat2, stat3, stat5 = 80, 70 - defenders_list[0][1], max(110, defenders_list[0][1]*2.5)
        pdict[attacker].stat12[enum] += max(70, defenders_list[0][1]*2) * dstrength
        pdict[attacker].stat13[enum] += max(110, defenders_list[0][1]*3) * dstrength
        pdict[attacker].stat1_unadj[enum] += max(110, defenders_list[0][1]*3)
    else:
        stat2, stat3, stat5 = 90, 50, 180
        pdict[attacker].stat11[enum] += 70 * dstrength
        pdict[attacker].stat12[enum] += 140 * dstrength
        pdict[attacker].stat13[enum] += 210 * dstrength
        pdict[attacker].stat1_unadj[enum] += 210
    
    # update all successful defenders with values set in last step
    if(astrength > thres):
        if(place < 2):
            astrength -= (2-place) * 0.1
        for pla, time in defenders_list:
            pdict[pla].stat5[enum] += stat5 * astrength
            if(time == 60):
                pdict[pla].place[enum] += 3
                pdict[pla].stat2[enum] += stat2 * astrength
                pdict[pla].stat2_unadj[enum] += stat2
                pdict[pla].stat3[enum] += stat3 * astrength
                pdict[pla].stat3_unadj[enum] += stat3
                pdict[pla].stat4[enum] += 1  
    return

def fill_pdict(nc=False, eve=None):
    pdict = {}

    mypath = os.path.dirname(__file__)

    if(not nc):
    # initialize all player datasets with 0s if they appear in the event, and -1s otherwise
        for i, ros in enumerate(all_rosters):
            for team in ros:
                for pla in team:
                    if(pla not in pdict):
                        pdict[pla] = Player()
                        for var in vars(pdict[pla]):
                            getattr(pdict[pla], var).extend([-1] * len(all_rosters))

                    # fill current event with 0s
                    for var in vars(pdict[pla]):
                        getattr(pdict[pla], var)[i] = 0
        
        txtfile = os.path.join(mypath,"places.txt")

    else:
        # for non-counted events, handle just the selected event
        for team in nc_roster[nc_nums[eve]]:
            for pla in team:
                pdict[pla] = Player()
                # fill current event with 0s
                for var in vars(pdict[pla]):
                    getattr(pdict[pla], var).append(0)
        
        txtfile = os.path.join(mypath,"places_uncounted.txt")
    
    f = open(txtfile, "r")
    text = f.read()
    f.close()

    # adjust imperfect data source to remove double and trailing spaces
    text = text.replace("  ", " ")
    text = text.replace(" \n", "\n")

    # split into events and iterate through each event
    events = text.split("\n===\n")
    average = []
    for enum, event in enumerate(events):
        # if nc, select only requested event and remove the header
        if(nc):
            if(event[:event.index("\n")] != eve):
                continue
            event = event[event.index("\n")+1:]
            enum = 0

        rounds = event.split("\n---\n")
        # 9 rounds of matches
        for j in range(round(len(rounds))):
            matches = rounds[j].split("\n\n")
            # in each round, 5 1v1 matches among the 10 teams
            for match in matches:
                lines = match.split("\n")

                # get the names of the attackers from the first line
                values = lines.pop(0).split(" ")
                home_stat1, away_stat1 = values[5], values[7]
                pdict[home_stat1].num_stat1[enum] += 1
                pdict[away_stat1].num_stat1[enum] += 1
                
                # pop summary (last line) from remaining lines
                summary = lines.pop()

                hdefenders_list = []
                adefenders_list = []
                for line in lines:
                    # skip redundant lines
                    if("nerojowa moeew krep" in line or "owf n jdofz jep" in line):
                        continue
                    
                    # extract the attacker, defender, and time from the text format
                    values = line.split(" ")
                    attacker = values[5]
                    defender = values[1]
                    time = int(values[0][3:5])

                    # store the info in the relevant attacker list
                    if(attacker == home_stat1):
                        hdefenders_list.append([defender, time])
                    else:
                        adefenders_list.append([defender, time])
                    
                # remove all commas for consistency, then split
                summary = summary.replace(",", "")
                summary = summary.split(" ")

                # if there are surviving defenders, deal with them
                if(summary[5] != "None"):
                    # list is ordered, team 1 survivors then team 2
                    num_a = 3 - len(adefenders_list)
                    for i in range(num_a):
                        adefenders_list.append([summary[5+i], 60])
                    for i in range(len(summary)-1, 4+num_a, -1):
                        hdefenders_list.append([summary[i], 60])
                
                # now process results for both halves of the match
                load_match(pdict, enum, average, home_stat1, hdefenders_list)
                load_match(pdict, enum, average, away_stat1, adefenders_list)

        for pla in pdict:
            if(pdict[pla].num_stat1[enum] != -1):
                # if player never attacked, set values to placeholder, otherwise divide by num_attacks
                attack = pdict[pla].num_stat1[enum] == 0
                for var in ["stat11", "stat12", "stat13", "stat1_unadj"]:
                    getattr(pdict[pla], var)[enum] = -1 if attack else getattr(pdict[pla], var)[enum] / pdict[pla].num_stat1[enum]

                # divide defend totals by number of defend rounds
                for var in ["stat2", "stat3", "place", "stat5", "stat3_unadj", "stat2_unadj"]:
                    getattr(pdict[pla], var)[enum] /= pdict[pla].stat4[enum]
        
        # compute average times for the event, used to normalize by event
        average.append(calculate_average(pdict,enum))
    
    # normalize all performances by the corresponding event averages
    for pla in pdict:
        for i in range(len(pdict[pla].place)):
            if(pdict[pla].place[i] != -1):
                pdict[pla].stat3[i] /= average[i][0]
                pdict[pla].stat2[i] /= average[i][2]
                pdict[pla].stat5[i] /= average[i][2]
                if(pdict[pla].stat13[i] != -1):
                    pdict[pla].stat13[i] /= average[i][1]
    
    if(nc):
        return pdict
    
    for pla in pdict:
        # determine number of previous events played in the last 7 for every event (plus current status)
        pdict[pla].num_last7.append(-1)
        for i in range(1, len(pdict[pla].num_last7)):
            eve_tot = 0
            # check last 7 events, or until start of event history
            for k in range(max(0, i-7), i):
                    if(pdict[pla].stat2[k] != -1):
                        eve_tot += 1
            # for early events, manually set to 2 if played in 1
            if(i < 3 and eve_tot == 1):
                eve_tot = 2
            pdict[pla].num_last7[i] = eve_tot
    
    # write an amalgamation of attack and defend stats for each player and event to a substitution file
    # used for other contests
    string = ""
    for i in range(len(pdict[pla].place)):
        for pla in pdict:
            if(pdict[pla].place[i] != -1):
                string += pla + "\t" + str(round(pdict[pla].stat2[i], 3)) + "\t" + str(round(1/pdict[pla].stat13[i], 3)) + "\n"
        string = string + "===\n"
    string = string[:-5]

    mypath = mypath[:-2] + "Exterior Stats"
    txtfile = os.path.join(mypath, "C7stats.txt")
    f = open(txtfile, 'w')
    f.write(string)
    f.close()
    
    # return result
    return pdict

def compute_team(pdict, loc, team, decay=0.85):
    # 3 best attackers are selected, need to store their attack scores
    stat1s = []
    z = []
    for pla in team:
        z.append(compute_eve_pla(pdict, loc, pla, decay, False))
        stat1s.append([z[-1].stat1 + z[-1].num_stat1/4, pla])
    
    # sort attack scores to determine order
    stat1s = sorted(stat1s, reverse=True)
    row = [0,0,0,0,0,0,0,0,0,0,0,0,0]
    for i in range(4):
        # if worst attack score, contributes exclusively to defend scores
        if(team[i] == stat1s[3][1]):
            row = np.add(row, [z[i].stat3 * 1/3, z[i].stat32 * 1/3, z[i].stat2 * 1/3, z[i].stat22 * 1/3, z[i].stat5 * 1/3,
                            0, 0, 0, 0, 0, 0, z[i].num_stat1, 0.25])
        else:
            # hard limit of how bad an attack can be evaluated at
            z[i].stat1 = max(0.2, z[i].stat1)
            # to compute combined scores, take weighted average of 3 other participants' defending
            sum_run_ind = 0
            sum_run_tot = 0
            for j in range(4):
                if(i != j):
                    sum_run_ind += z[j].stat3 / 3
                    sum_run_tot += z[j].stat2 / 3
            
            # compute combined scores (selected attacker * unselected defenders)
            ind_ratio = z[i].stat1 * sum_run_ind
            tot_ratio = z[i].stat1 * sum_run_tot

            # if 2nd worst attack score, slightly impacts attacking scores, greatly impacts defending
            if(team[i] == stat1s[2][1]):
                row = np.add(row, [z[i].stat3 * 8/27, z[i].stat32 * 8/27, z[i].stat2 * 8/27, z[i].stat22 * 8/27, z[i].stat5 * 8/27,
                                z[i].stat1 * 1/9, z[i].stat12 * 1/9, ind_ratio * 1/9, pow(ind_ratio, 2) * 1/9, 
                                tot_ratio * 1/9, pow(tot_ratio, 2) * 1/9, z[i].num_stat1, 0.25])
            # top 2, strongly impacts attack, less defense
            else:
                row = np.add(row, [z[i].stat3 * 5/27, z[i].stat32 * 5/27, z[i].stat2 * 5/27, z[i].stat22 * 5/27, z[i].stat5 * 5/27,
                                z[i].stat1 * 4/9, z[i].stat12 * 4/9, ind_ratio * 4/9, pow(ind_ratio, 2) * 4/9, 
                                tot_ratio * 4/9, pow(tot_ratio, 2) * 4/9, z[i].num_stat1, 0.25])
    
    return row

def train(pdict, eve_num=None, decay=0.85, d=default_d):
    roster_train = all_rosters[2:]

    if(eve_num is None):
        eve_num = len(roster_train) - 3

    loc = 1
    matrix = []

    # initialize stat priors for every team in every event
    for event in roster_train[:eve_num+3]:
        aver = 0
        for team in event:
            matrix.append(list(compute_team(pdict, loc, team, decay)))
            aver += matrix[-1][0] + matrix[-1][4]

        aver = aver / 40
        for j in range(10):
            matrix[-1 - j].append(aver)
        loc += 1

    # select specific stats for training
    matrix = np.delete(matrix, d, axis=1)

    # solving for the weight vector
    weights = [1000000]
    reg = 0
    # apply increasing regularization until weights are small enough (fights overfitting)
    while(max(weights) > 700 or min(weights) < -700):
            ident = np.identity(len(matrix[0])) * reg
            tmatrix0 = np.transpose(matrix)
            fmatrix0 = np.matmul(np.linalg.inv(np.matmul(tmatrix0, matrix) + ident), tmatrix0)
            weights = np.matmul(fmatrix0, results[:30+eve_num*10])
            reg += 0.3
    
    return weights

def predict(pdict, eve_num, decay=0.85, d=default_d, roster=None):
    weights = train(pdict, eve_num, decay, d)

    # if roster is supplied, override event #eve_num with roster
    if(roster is not None):
        predict_roster = roster
    else:
        predict_roster = all_rosters[eve_num+5]
    
    input_matrix = []
    average = 0

    # compute every team's stats going into the event
    for team in predict_roster:
        input_matrix.append(list(compute_team(pdict, eve_num+4, team, decay)))
        average += input_matrix[-1][0] + input_matrix[-1][4]

    # append average time of the whole event as another trainable variable
    average /= 40
    for j in range(10):
        input_matrix[j].append(average)
    
    # feature selection, delete stats numbered by d
    input_matrix = np.delete(input_matrix, d, axis=1)

    result = np.matmul(input_matrix, weights)

    norm = 9800 / sum(result)
    result = result * norm
    result = result.round(0)
        
    return result

def generate_pr(pdict, decay=0.85, d=default_d):
    weights = train(pdict, decay=decay, d=d)

    matrix_pr = []
    pr = []
    pdict_eve = {}

    # compute stats for every player to identify their individual skill level
    average = 0
    for pla in pdict:
        z = compute_eve_pla(pdict, len(all_rosters)-1, pla, decay=decay, ispr=True)
        pdict_eve[pla] = z
        # apply below replacement level attack stats to non-attackers
        if(z.num_stat1 < 1):
            z.stat1 = 0.3*(1 - z.num_stat1) + z.stat1*(z.num_stat1)
            z.num_stat1 = 1
        # smoothen attack freq
        z.num_stat1 = z.num_stat1 - (1/5)*(z.num_stat1 - 2.5)

        # compute ratios using implied "average" other half 
        # (attack stats * avg defend (1), and defense stats * avg attack (1))
        x = z.num_stat1
        y = 9 - x*0.75
        ind_ratio = (z.stat1 * x + z.stat3 * y / 3) / (x + y / 3)
        tot_ratio = (z.stat1 * x + z.stat2 * y / 3) / (x + y / 3)

        # create row for the player
        matrix_pr.append([z.stat3 * y / 27, z.stat32 * y / 27, z.stat2 * y / 27, z.stat22 * y / 27, z.stat5 * y / 27,
                          z.stat1 * x / 9, z.stat12 * x / 9, ind_ratio * (10+x*2) / 54,
                            pow(ind_ratio, 2) * (10+x*2) / 54, tot_ratio * (10+x*2) / 54,
                            pow(tot_ratio, 2) * (10+x*2) / 54, z.num_stat1, 0.25])

        average += z.stat3 + z.stat1
        pr.append([pla])
    
    # append averages (divide by number of players, then by 4 (share of team's))
    average = average / len(pdict)
    for row in matrix_pr:
        row.append(average/4)

    # compute scores and append to player names
    matrix_pr = np.delete(matrix_pr, d, axis=1)
    scores = np.matmul(matrix_pr, weights)
    for i in range(len(pr)):
        pr[i].append(scores[i])

    pr = (sorted(pr,key=lambda x: (x[1])))
    pr.reverse()

    # filter out players who aren't "current"
    f = open(os.path.join(os.path.split(os.path.dirname(__file__))[0], "current_list.txt"))
    text = f.read()
    lines = text.split("\n")
    pfreq_dict = {}
    for line in lines:
        values = line.split(" ")
        pfreq_dict[values[0]] = values[1]
    ind = 0
    while(ind < len(pr)):
        if(pr[ind][0] not in pfreq_dict.keys()):
            del pr[ind]
        else:
            ind += 1

    # get minimum score
    x = pr[len(pr) - 1][1]

    avg_scores = 0
    total = 0
    for player in pr:
        # ensure all scores are positive by linearly scaling all <30 scores to exist between 0-30
        if(x < 0 and player[1] < 30):
            player[1] = (player[1]-30) * (30/(30-x)) + 30

        avg_scores += player[1] * float(pfreq_dict[player[0]])
        total += float(pfreq_dict[player[0]])
    avg_scores /= total
    for player in pr:
        player[1] = round(player[1] * (240 / avg_scores))
    
    # reference variables for debugging
    stat1s = []
    for pla in pdict_eve:
        stat1s.append([pla, pdict_eve[pla].stat1])
    stat1s = (sorted(stat1s,key=lambda x: (x[1])))
    stat1s.reverse()

    stat2s = []
    for pla in pdict_eve:
        stat2s.append([pla, pdict_eve[pla].stat2])
    stat2s = (sorted(stat2s,key=lambda x: (x[1])))
    stat2s.reverse()  
    
    return pr

if __name__ == "__main__":    
    pdict = fill_pdict()

    cla = C7_class()
    pdict = cla.pdict

    perfList = []
    for key in cla.e_dict:
        if(key != "CUR"):
            val = cla.eve_pr(key)
            for value in val:
                perfList.append([value[1], value[0] + key[-2:]])
    perfList = sorted(perfList, reverse=True)

    # eve_pr test

    val = cla.eve_pr("EVE_43")

    # if current event, return player rankings
    if(cla.e_dict["CUR"] == eve_num):
        pr = generate_pr(pdict, decay=decay, d=default_d)
        string = ""
        for player in pr:
            string = string + (player[0] + "\t" + str(player[1]) + "\n")
        
        check = 0
    # otherwise, compute scores from selected event
    else:
        scores = predict(pdict, eve_num, decay, default_d)

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
        pr_preds_range = pr_preds[0+eve_num*10:10+eve_num*10]
        for j in range(10):
            error2 += np.power((pr_preds_range[j] - results_test[j]), 2)
            error2m.append(pr_preds_range[j] - results_test[j])
        error2 = error2 / 10
        error2 = np.sqrt(error2)

        # evaluate mse accuracy of naive prediction method for comparison
        error3 = 0
        equal_values = [980,980,980,980,980,980,980,980,980,980]
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