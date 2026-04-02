import numpy as np
import os.path
from time import perf_counter

##############################################################################################

# Consistently Updated Globals and Classes

default_decay = 0.85
default_d = [6, 7, 9, 13, 14]

e_num = 19

r_coef = 1

# numbered events where this contest appeared
event_select = [2,3,4,5,6,7,8,9,10,13,16,17,18,19,20,22,23,26,27,28,33,35,38,40,41,42,43]

# average scores for every event
event_averages = [9447, 9891, 9092, 7772, 7499, 15481, 13776, 8106, 9852, 7407, 8335, 10717, 13541, 13808, 12393, 11878, 7967, 10993, 10390, 10500]

# training results, removes high variance artifacts and normalizes
# - Adjustments: add back spent points, multiply by 10700 / sum(event)
results_train = [1370, 906, 1220, 519, 1814, 1257, 948, 348, 1282, 1035, 874, 1061, 1475, 604, 1030, 995, 1400, 1278, 1043, 939,
    852, 1092, 872, 960, 1476, 799, 1080, 1131, 1181, 1255, 1676, 560, 561, 1316, 886, 924, 1482, 533, 1337, 1425,
    1259, 1523, 1448, 733, 474, 1564, 929, 1071, 973, 727, 1616, 1115, 1083, 1145, 1258, 1344, 1177, 125, 1128, 707,
    995, 1069, 920, 1091, 822, 2011, 1138, 865, 1102, 687, 1030, 1184, 897, 512, 1516, 780, 930, 1591, 1197, 1063,
    822, 923, 1346, 1347, 1442, 846, 1037, 801, 1115, 1021, 943, 1182, 863, 961, 919, 1047, 1067, 1528, 1215, 976,
    1499, 1374, 797, 1051, 770, 692, 1367, 767, 753, 1625, 1444, 1172, 880, 727, 1039, 1150, 1054, 998, 1172, 1064,
    1361, 1476, 1074, 906, 625, 1541, 1130, 1197, 696, 694, 1100, 884, 1212, 1000, 911, 1100, 1205, 495, 1585, 1208,
    866, 1016, 1211, 1012, 1366, 1706, 1292, 421, 1325, 502, 741, 1306, 1126, 693, 1183, 1208, 1272, 1261, 759, 1147,
    1056, 1138, 981, 990, 1145, 1492, 1314, 1241, 897, 446, 791, 1508, 802, 883, 1127, 901, 1384, 1024, 1326, 954,
    943, 1243, 1305, 774, 1254, 992, 1302, 474, 919, 1494, 1318, 824, 813, 1951, 839, 1618, 824, 816, 639, 1058,
    1123, 1154, 1474, 327, 1146, 1248, 865, 943, 1091, 1330, 1512, 1230, 1018, 888, 1115, 527, 1193, 1081, 870, 1266
    ]

# actual results from event
results = [
    1344, 885, 668, 244, 1780, 1213, 784, 341, 1258, 1008, 1240, 1388, 2094, 518, 1462, 1409, 1988, 1783, 1230, 1051, 
    877, 1124, 898, 937, 1520, 792, 833, 1165, 809, 1293, 1432, 536, 137, 1260, 848, 885, 1419, 341, 1225, 1364, 
    1249, 1511, 1436, 727, 376, 1551, 881, 1062, 965, 133, 1568, 1082, 584, 1111, 1221, 1304, 320, 121, 1095, 686,
    847, 910, 571, 910, 281, 1712, 969, 736, 396, 440, 375, 903, 655, 293, 1222, 418, 515, 1290, 966, 862,
    1220, 1369, 1998, 1972, 2101, 1255, 1524, 1189, 1654, 1199, 1245, 1561, 1139, 1269, 1214, 1382, 1409, 2018, 1604, 1289,
    1279, 1172, 680, 816, 657, 368, 1166, 573, 278, 1117, 1460, 1185, 890, 735, 625, 1151, 759, 940, 1185, 922,
    1148, 720, 839, 764, 351, 740, 953, 1010, 587, 295, 1059, 851, 295, 720, 27, 1059, 1160, 476, 1525, 1163,
    866, 1016, 1211, 1012, 1366, 1706, 1292, 421, 1325, 502, 1027, 1809, 1560, 805, 1639, 1674, 1762, 1142, 533, 1590,
    1363, 967, 1266, 1278, 1452, 1925, 1600, 1295, 1113, 527, 993, 1914, 945, 1121, 1431, 1039, 1664, 970, 1447, 869,
    1047, 1380, 1449, 859, 1392, 1101, 1445, 526, 1020, 1659, 1241, 765, 0, 1844, 332, 1530, 257, 749, 249, 1000,
    1242, 907, 1696, 251, 1319, 1419, 652, 720, 1256, 1531, 1095, 1364, 1136, 593, 806, 516, 1390, 1260, 812, 1418
    ]

# normalize actual event results to standard total per event
results2 = results.copy()
for i in range(int(len(results2)/10)):
    total = sum(results2[i*10:(i+1)*10])
    for j in range(10):
        results2[i*10+j] = round(results2[i*10+j]*10700/total)

pr_preds = [967,967,967,967,967,967,967,967,967,967,1254,1497,1344,689,1156,958,1067,751,1190,757,
        723,918,1170,774,1046,1212,1120,430,1117,620,760,679,1055,847,899,791,1126,633,889,1001,
        967,967,967,967,967,967,967,967,967,967,1189,813,1123,1271,991,959,783,850,726,934,
        878,1181,940,938,953,990,589,954,909,968,728,664,1121,1168,635,545,974,667,1116,1353,
        1467,1305,1019,787,830,813,1289,965,1241,1182,1112,1192,808,1036,996,1280,1020,868,1116,1004,
        974,1189,1116,1396,1433,1083,887,435,1477,1123,725,876,1044,1119,849,998,1412,824,899,377,
        641,949,994,845,946,943,972,977,1001,687,1297,1097,948,722,1111,1182,1132,888,1005,459,
        968,1137,1099,911,1295,1142,1359,1201,977,981,939,1051,1102,899,1494,759,1271,1062,1116,1530,
        967,967,967,967,967,967,967,967,967,967,1100,1124,1418,1113,1378,943,1614,1424,1324,1271,
        967,967,967,967,967,967,967,967,967,967
        ]

skips = {9: {"P000", "P059", "P023", "P060"}, 10: {"P067"}, 12: {"P001"},
         24: {"P046", "P011", "P226", "P202"}, 25: {"P001"}}

# train on spent points 1/3 deducted
results_train = np.asarray(results2)/3 + np.asarray(results_train)*2/3

player_subs = {"P135": "P035", "P004": "P062", "P047": "P057", "P099": "P099", "P097": "P097",
"P046": "P001", "P016": "P035", "P026": "P099", "P134": "P042", "P130": "P127", "P056": "P024", "P006": "P017",
"P034": "P062", "P126": "P035", "P008": "P096", "P041": "P029", "P021": "P008", "P136": "P127", "P024": "P035",
"P052": "P017", "P028": "P074", "P005": "P016", "P002": "P008", "P030": "P036", "P009": "P008",
"P063": "P099", "P133": "P008", "P020": "P046", "P059": "P061", "P060": "P035", "P078": "P053", "P050": "P011",
"P124": "P044", "P064": "P010", "P000": "P074", "P125": "P006", "P017": "P038", "P137": "P060", "P039": "P144",
"P073": "P056", "P065": "P001", "P015": "P056", "P025": "P028", "P040": "P032", "P027": "P010",
"P075": "P056", "P061": "P059", "P115": "P017", "P109": "P017", "P132": "P017", "P106": "P062", "P055": "P055", 
"P023": "P019", "P010": "P017", "P100": "P038", "P012": "P025", "P131": "P017", "P144": "P059", "P038": "P017",
"P068": "P045", "P069": "P038", "P070": "P006", "P071": "P039", "P072": "P065", "P076": "P030", "P077": "P056",
"P141": "P027", "P143": "P030", "P200": "P032", "P118": "P026", "P098": "P067", "P094": "P030", "P203": "P030",
"P202": "P015", "P205": "P077", "P204": "P076", "P178": "P047", "P179": "P026", "P142": "P049", "P150": "P026",
"P168": "P035", "P117": "P006", "P107": "P006", "P079": "P051", "P096": "P127", "P210": "P030", "P213": "P016",
"P223": "P032", "P224": "P054", "P225": "P097", "P105": "P056", "P226": "P030", "P227": "P055",
"P231": "P019", "P149": "P040", "P139": "P015", "P214": "P036", "P140": "P026", "P232": "P077", "P233": "P032",
"P208": "P139", "P186": "P064"}

class C8_class:
    def __init__(self):
        self.pdict = fill_pdict()
        self.eve_str = compute_eve_str(self.pdict)
        self.e_dict = {"EVE_3": -8, "EVE_4": -7,"EVE_5": -6, "EVE_6": -5, "EVE_7": -4, "EVE_8": -3,
                    "EVE_9": -2, "EVE_10": -1, "EVE_11": 0, "EVE_14": 1, "EVE_18": 2, "EVE_19": 3, 
                    "EVE_21": 5, "EVE_22": 6, "EVE_24": 7, "EVE_25": 8, "EVE_28": 9, "EVE_29": 10, 
                    "EVE_30": 11, "EVE_35": 12, "EVE_37": 13, "EVE_40": 14, "EVE_42": 15, "EVE_43": 16,
                    "EVE_44": 17, "EVE_45": 18, "CUR": 19}
    # returns the predicted result of a previous event, with option to override with new roster
    def prev_sim(self, event, roster=None):
        results = predict(self.pdict, self.eve_str, self.e_dict[event], roster=roster)
        return results
    # returns player ranking results
    def pr_results(self):
        pr = generate_pr(self.pdict, self.eve_str)
        return pr
    # returns predicted result of a new event, occuring at current time
    def cur_sim(self, roster):
        results = predict(self.pdict, self.eve_str, self.e_dict["CUR"], roster=roster)
        results_indiv = []
        for val in results:
            for i in range(4): results_indiv.append(round(val/4))
        return results_indiv
    # computes and returns single event player scores
    def eve_pr(self, event):
        ind = min(max(5, self.e_dict[event]+3), self.e_dict["CUR"])
        weights = train(self.pdict, self.eve_str, eve_num=ind, d=[7, 8, 9, 13, 14])
        if(event in ["EVE_17", "EVE_23"]):
            # need the uncounted data for the above events
            dict_nc = fill_pdict(True, event)
            return eve_pr_calc(dict_nc, -8, weights, d=[7, 8, 9, 13, 14])

        return eve_pr_calc(self.pdict, self.e_dict[event], weights, d = [7, 8, 9, 13, 14])

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
all_rosters[0][1][2] = "P042"

# player class storing stats events lists for each stat
class Player:
    def __init__(self, num_eve):
        self.stat9s = [-1] * num_eve
        self.vstat9s = [-1] * num_eve
        self.stat2 = [-1] * num_eve
        self.stat4 = [-1] * num_eve
        self.stat3 = [-1] * num_eve
        self.time = [-1] * num_eve
        self.stat9_rate = [-1] * num_eve
        self.stat7 = [-1] * num_eve
        self.stat6 = [1] * num_eve
        self.stat5 = [-1] * num_eve
        
        self.num_last7 = [-1] * num_eve

# player class storing stats going into a particular event
class eventPlayer:
    def __init__(self):
        self.stat9s = 0
        self.stat9s2 = 0
        self.vstat9s = 0
        self.vstat9s2 = 0
        self.stat4 = 0
        self.stat2 = 0
        self.time = 0
        self.time2 = 0
        self.stat9_rate = 0
        self.stat9_rate2 = 0
        self.stat6 = 0
        self.stat62 = 0
        self.stat5 = 0
        self.stat52 = 0
        self.stat7 = 0

        self.stat3 = 0

        self.s_stat9s = 0
        self.s_stat9s2 = 0
        self.s_stat9_rate = 0
        self.s_stat9_rate2 = 0
        self.s_stat5 = 0
        self.s_stat52 = 0


def eve_pr_calc(pdict, eve, weights, d):
    loc = eve + 8
    matrix = []
    plist = []

    for pla in pdict:
        if(pdict[pla].stat9s[loc] == -1):
            continue
        
        # if not a one-off event, use previous event's r values
        if(len(pdict[pla].stat9s) > 1):
            stat2 = 0
            stat4 = 0
            inc = loc-1
            count = 0
            total = 0

            # compute stat4 over last 6 events played
            while(inc >= 0 and count < 6):
                # skip outlier results
                if(inc in skips and pla in skips[inc]):
                    inc -= 1
                    continue
                # linear decay over 6 events
                if(pdict[pla].stat4[inc] != -1):
                    stat2 += pdict[pla].stat2[inc] * (6-count)
                    stat4 += pdict[pla].stat4[inc] * (6-count)
                    total += (6-count)
                    count += 1
                inc -= 1
            
            stat2 /= max(total, 1)
            stat4 /= max(total, 1)
            # if played less than 6 events, normalize results towards average score
            if(count < 6):
                stat2 = stat2*total/27 + 0.02*(27-total)/27
                stat4 = stat4*total/27 + 0.02*(27-total)/27

            # use stat4 from previous events and current event, 6:4 ratio
            factor = (stat2 + stat4) * 0.6 + (pdict[pla].stat2[loc] + pdict[pla].stat4[loc]) * 0.4
        else:
            # if one off event, lower r for event
            factor = (pdict[pla].stat2[loc] + pdict[pla].stat4[loc]) * 0.6

        r = 1 - factor
        z = pdict[pla]

        # for stat3, some results are 0ed out
        if(z.stat3[loc] == 1):
            matrix.append([z.stat9s[loc]*r, pow(z.stat9s[loc], 2)*r, z.stat5[loc], pow(z.stat5[loc], 2), 
                        0, 0, z.time[loc], pow(z.time[loc], 2),
                       z.stat6[loc], pow(z.stat6[loc], 2), z.stat9_rate[loc]*r, pow(z.stat9_rate[loc], 2)*r, 0, r, 1, 1])
        else:
            matrix.append([z.stat9s[loc]*r, pow(z.stat9s[loc], 2)*r, z.stat5[loc], pow(z.stat5[loc], 2), 
                        z.vstat9s[loc]*r, pow(z.vstat9s[loc], 2)*r, z.time[loc], pow(z.time[loc], 2),
                        z.stat6[loc], pow(z.stat6[loc], 2), z.stat9_rate[loc]*r, pow(z.stat9_rate[loc], 2)*r, 
                        z.stat7[loc], r, 1, 1])
        plist.append([pla])
    
    matrix = np.delete(matrix, d, axis=1)

    # compute the scores, normalize
    vals = np.matmul(matrix, weights)

    # remove major outlier team if specific event
    if(eve == 16):
        vals = list(vals)
        indexes = sorted([plist.index(["P046"]), plist.index(["P011"]), plist.index(["P226"]), plist.index(["P202"])], reverse=True)
        for ind in indexes: 
            del vals[ind]
            del plist[ind]
        vals = np.array(vals)
        tot = sum(vals)
        vals *= 9000/tot
    else:
        tot = sum(vals)
        vals *= 10000/tot

    for i in range(len(plist)):
        plist[i].append(float(round(vals[i])))

    # sort by score and return
    plist = (sorted(plist,key=lambda x: (x[1])))
    plist.reverse()
    return plist

def build_player(pdict, loc, pla, eve_str, decay=default_decay):
    inc = loc
    depth = 0
    div = [0,0,0,0,0,0]
    outlier1 = [[],[],[],[]]
    outlier2 = []
    event_player = eventPlayer()

    # stat6 is precomputed for each event, select stat6[inc]
    event_player.stat6 = pdict[pla].stat6[inc]
    event_player.stat62 = pow(pdict[pla].stat6[inc], 2)

    while(inc >= 0):
        # skip clear outlier performances
        if(inc in skips and pla in skips[inc]):
            inc -= 1
            depth += 1
            continue
    
        value = pdict[pla].time[inc]

        # if player did not play in this event, skip
        if value != -1:
            # time, stat4, stat2, stat3 all handled, don't depend on stat3
            value *= eve_str[inc]
            event_player.time += value * pow(decay, depth)
            event_player.time2 += pow(value,2) * pow(decay, depth)
            div[0] += pow(decay,depth)
            
            # reduce decay used for stat4 and stat2, higher variance stat
            value = pdict[pla].stat4[inc]
            event_player.stat4 += value * pow((decay-1)/2+1, depth)
            event_player.stat2 += pdict[pla].stat2[inc] * pow((decay-1)/2+1, depth)
            div[1] += pow((decay-1)/2+1, depth)

            value = pdict[pla].stat3[inc]
            event_player.stat3 += value * pow(decay, depth)
            div[2] += pow(decay,depth)

            # if player isn't stat3, compute non-stat3 stats
            if(pdict[pla].stat3[inc] == 0):
                value = pdict[pla].stat9s[inc]
                value *= eve_str[inc]

                event_player.stat9s += value * pow(decay, depth)
                event_player.stat9s2 += pow(value,2) * pow(decay, depth)
                div[3] += pow(decay,depth)
                if(depth < 5):
                    outlier1[0].append(value * pow(decay, depth))
                    outlier2.append(depth)
                
                value = pdict[pla].stat9_rate[inc]
                value *= eve_str[inc]

                event_player.stat9_rate += value * pow(decay, depth)
                event_player.stat9_rate2 += pow(value,2) * pow(decay, depth)
                if(depth < 5):
                    outlier1[1].append(value * pow(decay, depth))

                value = pdict[pla].vstat9s[inc]
                value *= eve_str[inc]

                # reduce decay used for vstat9s, higher variance stat
                event_player.vstat9s += value * pow((decay-1)/2+1, depth)
                event_player.vstat9s2 += pow(value,2) * pow((decay-1)/2+1, depth)
                div[4] += pow((decay-1)/2+1,depth)

                value = pdict[pla].stat5[inc]
                value *= eve_str[inc]

                event_player.stat5 += value * pow(decay, depth)
                event_player.stat52 += pow(value,2) * pow(decay, depth)
                if(depth < 5):
                    outlier1[2].append(value * pow(decay, depth))
                
                value = pdict[pla].stat7[inc]
                value *= eve_str[inc]

                event_player.stat7 += value * pow(decay, depth)
                if(depth < 5):
                    outlier1[3].append(value * pow(decay, depth))
                    
            # player is stat3, compute stat3 stats
            else:
                value = pdict[pla].stat9s[inc]
                if value != -1:
                    event_player.s_stat9s += value * pow(decay, depth)
                    event_player.s_stat9s2 += pow(value,2) * pow(decay, depth)
                    div[5] += pow(decay,depth)

                value = pdict[pla].stat9_rate[inc]
                if value != -1:
                    event_player.s_stat9_rate += value * pow(decay, depth)
                    event_player.s_stat9_rate2 += pow(value,2) * pow(decay, depth)

                value = pdict[pla].stat5[inc]
                event_player.s_stat5 += value * pow(decay, depth)
                event_player.s_stat52 += pow(value,2) * pow(decay, depth)

        
        inc -= 1
        depth += 1
        # double increase at inc=18, representing more time passing
        if(inc in [18]):
            depth += 1
    
    # do outlier analysis if more than 3 events
    num_e = len(outlier2)
    if(num_e > 3):
        full_values = [(outlier1[0][i]+outlier1[2][i]) / pow(decay, outlier2[i]) for i in range(num_e)]
        mean = np.mean(full_values)
        std = np.std(full_values)
        for k in range(len(full_values)):
            # reduce outlier's effect by 50%, for squared values need to compute first
            if(abs((full_values[k] - mean)/std) > 1.7):
                for i, var in enumerate(["stat9s", "stat9s2", "stat9_rate", "stat9_rate2", "stat5", "stat52", "stat7"]):
                    if(i % 2 == 0):
                        setattr(event_player, var, vars(event_player)[var] - outlier1[i//2][k] * 0.5)
                    else:
                        setattr(event_player, var, vars(event_player)[var] - pow(outlier1[i//2][k], 2) * 0.5 / pow(decay, outlier2[k]))
                div[3] -= pow(decay, outlier2[k]) * 0.5
    
    # divide by sum of the weights for non-stat3 stats (if present)
    if(div[3] != 0):
        for var in ["stat9s", "stat9s2", "stat9_rate", "stat9_rate2", "stat5", "stat52", "stat7"]:
            setattr(event_player, var, getattr(event_player, var) / div[3])
        for var in ["vstat9s", "vstat9s2"]:
            setattr(event_player, var, getattr(event_player, var) / div[4])

    # slight normalization factor
    event_player.stat2 += 0.1 * pow(decay, 2)
    event_player.stat4 += 0.1 * pow(decay, 2)
    div[1] += pow(decay, 2)

    # divide by sum of weights for overall stats
    event_player.stat2 /= div[1]
    event_player.stat4 /= div[1]
    event_player.time /= div[0]
    event_player.time2 /= div[0]
    event_player.stat3 /= div[2]

    # divide by sum of weights for stat3 stats
    if(div[5] != 0):
        for var in ["s_stat9s", "s_stat9s2", "s_stat9_rate", "s_stat9_rate2", "s_stat5", "s_stat52"]:
            setattr(event_player, var, getattr(event_player, var) / div[5])

    return event_player

def compute_eve_pla(pdict, loc, pla, eve_str, decay, ispr):
    # if player not initialized or (haven't played many recent events and doing predictions)
    if(pla not in pdict or (pdict[pla].num_last7[loc+1] < 2 and not ispr)):
        # determine player substitution values
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

def fill_pdict(nc=False, event_name=None):

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

    if(not nc):
        # export stat sheet to exterior stats, used for other models
        mypath = mypath[:-2] + "Exterior Stats"
        txtfile = os.path.join(mypath, "C8stats.txt")
        f = open(txtfile, 'w')
        f.write(text)
        f.close()

    events = text.split("\n===\n")

    time_results = []
    # read in all stats
    for enum, event in enumerate(events):
        lines = event.split("\n")
        # for non-counted events, only read in the event with the relevant header
        if(nc):
            if(lines[0] != event_name):
                continue
            lines = lines[1:]
            enum = 0

        pla_list = []
        tot_discount = 0
        average = [0,0,0,0,0]
        for i in range(len(lines)):
            # read in all values line by line, including computing ratios
            values = lines[i].split("\t")

            name = values[0]
            if(name not in pdict):
                # if not counted, initialize player with 1 entry per stat, else number of events
                pdict[name] = Player(1 if nc else len(events))

            # team by team, save team's time by reading every 4 lines
            if(i % 4 == 0):
                time_results.append(float(values[6]))

            pla_list.append(name)

            pdict[name].stat9s[enum] = int(values[2])
            average[0] += int(values[2])
            pdict[name].vstat9s[enum] = int(values[3])
            # normalize averages for vstat9s by adding baseine value
            average[1] += int(values[3]) + 40

            pdict[name].stat4[enum] = float(values[4]) / (float(values[2]) + float(values[3]))
            pdict[name].stat3[enum] = int(values[1])
            pdict[name].stat5[enum] = float(values[7]) - float(values[8]) / 2
            pdict[name].stat7[enum] = (float(values[7]) - float(values[8]) / 2) / float(values[6])
            pdict[name].stat9_rate[enum] = float(values[2]) / float(values[6])

            # adjust stats from outlier event
            if(enum == 22 and name in ["P042", "P227", "P224", "P059"]):
                for var in ["stat9s", "stat5", "stat7", "stat9_rate"]:
                    getattr(pdict[name], var)[enum] = int(getattr(pdict[name], var)[enum] * 0.85)

            # stat3 and non-stat3 have different stat5 averages
            if(int(values[1]) == 0):
                # for enum == 4, certain performances are not included for computing stat5 average
                if(enum != 4 or i not in [4,5,6,34,33,32]):
                    average[3] += int(values[7])
            else:
                average[4] += int(values[7])
            
            # determine how much of stat2 is individual and how much is shared to the team
            if(values[1] == "1"):
                pers_coef = 1
                shared_coef = 0
            else:
                pers_coef = 1/2
                shared_coef = 1/8
            
            # add personal weighed "stat2" to stats, and shared result to team's total
            pdict[name].stat2[enum] = float(values[5])/ (float(values[2]) + float(values[3])) * pers_coef
            tot_discount += float(values[5])/(float(values[2]) + float(values[3]))  * shared_coef

            pdict[name].time[enum] = float(values[6])
            average[2] += float(values[6])

            # if last player, add shared values to stat2 for every player on team
            if(i % 4 == 3):
                for player in pla_list:
                    pdict[player].stat2[enum] = min(pdict[player].stat2[enum]+tot_discount, 1)
                pla_list = []
                tot_discount = 0

        # compute averages for all stats by dividing by total     
        for i in range(3):
            average[i] /= 40
        if(enum != 4):
            average[3] /= 30
        else:
            average[3] /= 24
        average[4] /= 10

        # adjust time_results for training stat6 later
        for i in range(10):
            time_results[enum*10 + i] /= average[2]
            time_results[enum*10 + i] -= 1
        
        for pla in pdict:
            if pdict[pla].stat9s[enum] != -1:
                # normalize all results using averages
                pdict[pla].stat9s[enum] /= average[0]
                pdict[pla].vstat9s[enum] /= average[1]
                pdict[pla].time[enum] /= average[2]
                if(pdict[pla].stat3[enum] == 0):
                    pdict[pla].stat5[enum] /= average[3]
                else:
                    pdict[pla].stat5[enum] /= average[4]
                pdict[pla].stat9_rate[enum] /= (average[0] / average[2])
                pdict[pla].stat7[enum] /= (average[3] / average[2])
    
    # if not counted, can return without backfill or stat6 computation
    if(nc):
        for pla in pdict:
            pdict[pla].stat6 = [1]
        return pdict
    
    ### compute stat6 results for every player after every event

    pindex = {}
    index = 0
    all_rosters[20][8][0] = "P059"
    l = len(all_rosters)
    w = len(pdict)

    # team player matrix, each row representing a team, each column a player
    # full_matrix[i][j] != 0 means player j is on team i
    full_matrix = np.zeros((l*10, w))

    # reduce magnitude of decay, then invert
    decay = 2/(default_decay+1)

    adj_results = np.copy(time_results)
    pindex["P099"] = [-1, -1]

    for i in range(len(all_rosters)):
        # assign each returning player an index and populate full_matrix with appearances
        for j, team in enumerate(all_rosters[i]):
            # removing outlier teams
            if(i == 9 and j == 4):
                continue
            if(i == 10 and j == 7):
                continue
            for pla in team:
                if(pla == "P099"):
                    # outlier player
                    adj_results[i*10+j] += 0.1
                elif(pla not in pindex):
                    # first appearance, no index assigned, store row to assign upon next appearance
                    pindex[pla] = [-1, i*10+j]
                    adj_results[i*10+j] += 0.015
                elif(pindex[pla][0] == -1):
                    # second appearance, initialize index, set both current row and prev appearance row
                    pindex[pla][0] = index
                    full_matrix[i*10+j][index] = pow(decay, i)
                    prev = pindex[pla][1]
                    full_matrix[prev][index] = pow(decay, prev//10)
                    # partially undo adjustment to prev target, not a one time player
                    adj_results[prev] -= 0.01 * pow(decay, prev//10)
                    index += 1
                else:
                    # 3rd or more, get previously assigned index and set column of row
                    ind = pindex[pla][0]
                    full_matrix[i*10+j][ind] = pow(decay, i)
            adj_results[i*10+j] *= pow(decay, i)
        
        # need at least 5 events to get reasonable results
        if(i < 5):
            continue

        # select just the assigned portion of full_matrix
        train_matrix = full_matrix[:(i+1)*10, :index]

        # estimate "skill" levels for each player using the constructed sparse matrix
        # apply increasing regularization until results are reasonable (useful for earlier events)
        weights = [1]
        incr = pow(max(train_matrix[-2]) / 2, 2)
        reg = incr
        while(max(weights) > 0.125 or min(weights) < -0.125):
            ident = np.identity(len(train_matrix[0])) * reg
            tmatrix0 = np.transpose(train_matrix)
            fmatrix0 = np.matmul(np.linalg.inv(np.matmul(tmatrix0, train_matrix) + ident), tmatrix0)
            weights = np.matmul(fmatrix0, adj_results[:(i+1)*10])
            reg += incr
        
        # overwrite stat6 values for every player using the derived "skill" levels
        for pla in pindex:
            if(pindex[pla][0] != -1):
                pdict[pla].stat6[i] = weights[pindex[pla][0]] * 4 + 1
            elif(pla == "P099"):
                pdict[pla].stat6[i] = 0.65
            else:
                pdict[pla].stat6[i] = 0.98
    
    # revert change to roster
    all_rosters[20][8][0] = "P000"
    
    for pla in pdict:
        # determine number of previous events played in the last 7 for every event (plus current status)
        pdict[pla].num_last7.append(-1)
        for i in range(1, len(pdict[pla].num_last7)):
            eve_tot = 0
            # check last 7 events, or until start of event history
            for k in range(max(0, i-7), i):
                    if(pdict[pla].time[k] != -1):
                        eve_tot += 1
            # for early events, manually set to 2 if played in 1
            if(i < 2 and eve_tot == 1):
                eve_tot = 2
            pdict[pla].num_last7[i] = eve_tot

    return pdict

def compute_eve_str(pdict, decay=default_decay):
    eve_str = [1] * 100
    new_eve_str = []
    for i in range(len(pdict["P053"].time)):
        new_eve_str.append(0)
        # only compute strength for events 9 and higher
        if(i >= 5):
            roster = all_rosters[i]
            # sum stat9s average going into the event for all players
            for team in roster:
                for pla in team:
                    eve_pla = compute_eve_pla(pdict, i-1, pla, eve_str, decay, False)
                    new_eve_str[-1] += eve_pla.stat9s
        else:
            new_eve_str[-1] = 1.0
    
    # find average over all events for normalization
    aver = np.mean(new_eve_str[5:])

    # normalize and scale all strengths
    for i in range(5, len(new_eve_str)):
        new_eve_str[i] = round(new_eve_str[i] / aver, 4)

    # set final (current) event to 1, neutral strength
    new_eve_str.append(1)

    return new_eve_str

def compute_team(pdict, loc, team, eve_str, decay=default_decay):
    p_list = []
    # instantiate stats for each player
    for pla in team:
        p_list.append([pla, compute_eve_pla(pdict, loc, pla, eve_str, decay, False)])
    
    # sort to find the most common stat3s, tie break with overall success
    p_list = sorted(p_list, key=lambda x: (-x[1].stat3, x[1].stat9s))

    # if freq results are close and large success gap, swap
    if(p_list[0][1].stat3 - p_list[1][1].stat3 < 0.5 and p_list[1][1].stat9s > p_list[0][1].stat9s*1.1):
        p_list[0], p_list[1] = p_list[1], p_list[0]
    
    # if no player has stat3 experience, use standard stats
    if(p_list[0][1].stat3 == 0):
        p_list[0][1] = compute_eve_pla(pdict, loc, "P053", eve_str, decay, False)
    
    # if two players have no non-stat3 experience, use standard stats
    if(p_list[1][1].stat3 == 1):
        p_list[0][1] = compute_eve_pla(pdict, loc, "P017", eve_str, decay, False)
    
    row = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    # add stats for stat3
    z = p_list[0][1]
    row = np.add(row, [z.s_stat9s, z.s_stat9s2, z.s_stat5, z.s_stat52, 0, 0, z.time, 
        z.time2, z.stat6, z.stat62, z.s_stat9_rate, z.s_stat9_rate2, 0, 1, 1])
    
    # add stats for non-stat3s
    for i in range(1, 4):
        z = p_list[i][1]
        r = 1 - (z.stat2 + z.stat4) * r_coef
        row = np.add(row, [z.stat9s*r, z.stat9s2*r, z.stat5, z.stat52, 
            z.vstat9s*r, z.vstat9s2*r, z.time, z.time2, z.stat6, 
            z.stat62, z.stat9_rate*r, z.stat9_rate2*r, z.stat7, r, 1])

    row = np.divide(row, 4)

    return row

def train(pdict, eve_str, eve_num=None, decay=0.85, d=default_d):
    roster_train = all_rosters[5:]

    if(eve_num is None):
        eve_num = len(roster_train) - 3

    loc = 4
    matrix = []

    for event in roster_train[:eve_num+3]:
        aver = 0
        # for every team, initialize a matrix row containing the stat averages for the team
        for team in event:
            row = compute_team(pdict, loc, team, eve_str, decay)
            aver += row[0]
            matrix.append(list(row))
        
        # append average stat9s of the whole event as another trainable variable
        aver = aver / 10
        for j in range(10):
            matrix[-1 - j].append(aver)
        loc += 1

    # feature selection, delete stats numbered by d
    matrix = np.delete(matrix, d, axis=1)

    # solve linear regression problem using matrix algebra
    # limit magnitude of weights to reasonable threshold, increase regularization
    # until threshold is met
    weights = [1000000]
    reg = 0
    while(max(weights) > 200 or min(weights) < -200):
            ident = np.identity(len(matrix[0])) * reg
            tmatrix0 = np.transpose(matrix[0:30+eve_num*10])
            fmatrix0 = np.matmul(np.linalg.inv(np.matmul(tmatrix0, matrix[0:30+eve_num*10]) + ident), tmatrix0)
            weights = np.matmul(fmatrix0, results_train[0:30+eve_num*10])
            reg += 2
    
    return weights

def predict(pdict, eve_str, eve_num, decay=0.85, d=default_d, roster=None):
    weights = train(pdict, eve_str, eve_num, decay, d)

    # if roster is supplied, override event #eve_num with roster
    if(roster is not None):
        predict_roster = roster
    else:
        predict_roster = all_rosters[eve_num+8]
    
    input_matrix = []
    average = 0

    # compute every team's stats going into the event
    for team in predict_roster:
        input_matrix.append(list(compute_team(pdict, eve_num+7, team, eve_str, decay)))
        average += input_matrix[-1][0]

    # append average of the whole event as another trainable variable
    average /= 10
    for j in range(10):
        input_matrix[j].append(average)
    
    # feature selection, delete stats numbered by d
    input_matrix = np.delete(input_matrix, d, axis=1)

    result = np.matmul(input_matrix, weights)

    norm = event_averages[eve_num] / sum(result)
    result = result * norm
    result = result.round(0)
        
    return result

def generate_pr(pdict, eve_str, decay=default_decay, d=default_d):
    weights = train(pdict, eve_str, decay=decay, d=d)

    matrix_pr = []
    pdict_eve = {}
    pr = []

    # compute stats for every player to identify their individual skill level
    average = 0
    for pla in pdict:
        z = compute_eve_pla(pdict, len(all_rosters)-1, pla, eve_str, decay=decay, ispr=True)
        pdict_eve[pla] = z
        # if player is stat3 rarely, only consider non-stat3 stats
        if(z.stat3 <= 0.33):
            z.stat3 = 0
        freqa = 1-z.stat3
        freqb = z.stat3
        r = 1 - (z.stat2 + z.stat4) * r_coef
        # compute weighted average of stat3 and non-stat3 stat
        val1 = [z.stat9s * freqa*r, z.stat9s2 * freqa*r, z.stat5 * freqa*r, z.stat52 * freqa*r,
                           z.vstat9s * freqa*r, z.vstat9s2 * freqa*r,
                             z.time, z.time2, z.stat6, z.stat62, z.stat9_rate * freqa*r,
                             z.stat9_rate2 * freqa*r, z.stat7 * freqa, r, 1]
        val2 = [z.s_stat9s * freqb, z.s_stat9s2 * freqb, z.s_stat5 * freqb, z.s_stat52 * freqb,
                0, 0, 0, 0, 0, 0, z.s_stat9_rate * freqb, z.s_stat9_rate2 * freqb, 0, 0, 0]
        result = list(np.add(val1, val2))

        matrix_pr.append(result)
        average += z.stat9s * (1-z.stat3) * r + z.s_stat9s * z.stat3
        pr.append([pla])
    
    # append averages (divide by number of players, then by 4 (share of team's))
    average = average / len(pdict)
    for row in matrix_pr:
        row.append(average)
    
    # compute the scores on all player stats using computed weights
    matrix_pr = np.delete(matrix_pr, d, axis=1)
    scores = np.matmul(matrix_pr, weights)
    for i in range(len(pr)):
        pr[i].append(scores[i])

    pr = (sorted(pr,key=lambda x: (x[1])))
    pr.reverse()

    # obtain the frequencies each player currently plays in the event
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

    # normalize scores such that the average of the playerbase is 250
    # weighted by the player freqs
    avg_scores = 0
    total = 0
    for player in pr:
        avg_scores += player[1] * float(pfreq_dict[player[0]])
        total += float(pfreq_dict[player[0]])
    avg_scores /= total
    for player in pr:
        player[1] = round(player[1] * (250 / avg_scores))
    
    return pr


if __name__ == "__main__":

    cla = C8_class()

    pdict = cla.pdict

    perfList = []
    for key in cla.e_dict:
        if(key != "CUR"):
            val = cla.eve_pr(key)
            for value in val:
                perfList.append([value[1], value[0] + key[-2:]])
    perfList = sorted(perfList, reverse=True)

    # test eve_pr
    val = cla.eve_pr("EVE_45")

    # if current event, return player rankings
    if(cla.e_dict["CUR"] == e_num):
        pr = generate_pr(pdict, cla.eve_str, default_decay, default_d)
        string = ""
        for player in pr:
            string = string + (player[0] + "\t" + str(player[1]) + "\n")
        
        check = 0
    else:
        scores = predict(pdict, cla.eve_str, e_num, default_decay, default_d)

        # for simulated events, check against true results for mse accuracy
        results_test = results[30+e_num*10:40+e_num*10]
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
        pr_preds_range = pr_preds[0+e_num*10:10+e_num*10]
        # normalize pr_preds to actual average for event
        tot = sum(pr_preds_range)
        for j in range(10):
            pr_preds_range[j] = round(pr_preds_range[j] * sum(results_test) / tot)
        for j in range(10):
            error2 += np.power((pr_preds_range[j] - results_test[j]), 2)
            error2m.append(pr_preds_range[j] - results_test[j])
        error2 = error2 / 10
        error2 = np.sqrt(error2)

        # evaluate error of naive guess vs true results
        error3 = 0
        equal_values = [int(event_averages[e_num]/10)] * 10 
        for j in range(10):
            error3 += np.power((equal_values[j] - results_test[j]), 2)
        error3 = error3 / 10
        error3 = np.sqrt(error3)

        errom = [int(x) for x in errorm]
        errom2 = [pow(x,2) for x in errom]
        error2m2 = [pow(x,2) for x in error2m]

        print(error)
        check = 0