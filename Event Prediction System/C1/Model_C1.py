import numpy as np
import os.path

##############################################################################################

# Consistently Updated Globals and Classes

decay = 0.9
default_d = []

eve_num = 15

# numbered events where this contest appeared
event_select = [14,16,17,19,22,23,25,27,29,31,32,35,37,39,40,41,42]

# player by player results for all events
results = [
    470, 340, 110, 30, 90, 180, 130, 440, 330, 360, 100, 80, 700, 200, 235, 120, 460, 630, 160, 350,
    480, 190, 60, 265, 560, 450, 245, 50, 490, 10, 40, 150, 370, 140, 20, 275, 70, 255, 170, 210,
    630, 200, 70, 255, 700, 130, 50, 150, 440, 350, 190, 170, 160, 450, 140, 235, 560, 210, 80, 60,
    480, 340, 370, 330, 30, 490, 100, 360, 190, 40, 10, 20, 470, 245, 110, 265, 275, 460, 90, 120,
    470, 370, 90, 20, 235, 490, 265, 100, 350, 245, 330, 50, 450, 360, 70, 180, 480, 190, 110, 80, 
    440, 210, 340, 160, 460, 275, 60, 255, 560, 170, 40, 120, 630, 130, 140, 30, 700, 200, 150, 10,
    360, 180, 130, 80, 490, 190, 90, 60, 470, 480, 160, 30, 350, 265, 150, 330, 630, 460, 140, 70,
    120, 560, 235, 110, 700, 340, 40, 10, 245, 450, 20, 50, 440, 275, 370, 200, 210, 100, 170, 255,
    700, 350, 170, 70, 180, 330, 370, 20, 630, 140, 210, 275, 480, 450, 235, 40, 470, 440, 160, 50,
    340, 200, 30, 110, 460, 245, 120, 10, 90, 265, 100, 130, 560, 255, 190, 80, 490, 360, 150, 60,
    460, 340, 70, 150, 480, 440, 100, 140, 630, 470, 330, 275, 490, 120, 20, 60, 350, 190, 210, 40, 
    450, 265, 10, 80, 370, 180, 200, 170, 245, 235, 130, 360, 700, 560, 110, 50, 160, 255, 90, 30,
    630, 80, 170, 60, 440, 275, 235, 10, 560, 330, 200, 190, 450, 490, 50, 100, 700, 350, 180, 20,
    340, 360, 255, 30, 480, 90, 140, 40, 265, 245, 70, 130, 460, 470, 110, 150, 370, 160, 210, 120,
    340, 265, 120, 60, 630, 330, 210, 30, 560, 360, 130, 50, 370, 150, 110, 160, 490, 440, 235, 70,
    450, 190, 245, 100, 460, 480, 470, 140, 170, 40, 20, 10, 700, 80, 200, 90, 350, 275, 255, 180,
    630, 210, 90, 40, 340, 360, 90, 150, 255, 275, 170, 110, 560, 350, 30, 20, 440, 490, 100, 0,
    480, 330, 190, 245, 700, 120, 200, 130, 140, 180, 265, 50, 470, 460, 70, 60, 370, 450, 235, 160,
    370, 440, 50, 40, 560, 150, 200, 90, 490, 450, 100, 60, 275, 265, 350, 170, 630, 210, 20, 30,
    460, 470, 235, 120, 480, 110, 190, 10, 330, 245, 80, 130, 340, 180, 360, 160, 700, 255, 70, 140,
    700, 490, 160, 60, 630, 130, 180, 50, 470, 265, 210, 90, 480, 190, 70, 120, 460, 255, 170, 10,
    340, 350, 245, 100, 440, 370, 330, 40, 450, 275, 360, 30, 200, 235, 150, 140, 560, 80, 110, 20,
    450, 440, 255, 30, 490, 330, 235, 150, 350, 180, 80, 170, 470, 275, 190, 120, 630, 245, 210, 130,
    560, 265, 110, 160, 480, 340, 70, 60, 460, 200, 40, 20, 700, 370, 140, 50, 360, 100, 90, 0,
    630, 340, 90, 150, 700, 130, 190, 200, 480, 440, 140, 160, 460, 265, 120, 80, 20, 30, 50, 40,
    360, 370, 275, 180, 560, 170, 110, 70, 330, 235, 255, 160, 490, 350, 245, 100, 470, 450, 60, 0,
    470, 190, 130, 0, 490, 340, 180, 20, 460, 235, 90, 80, 440, 30, 70, 60, 360, 210, 140, 100,
    700, 255, 110, 50, 560, 245, 150, 160, 275, 350, 265, 200, 370, 170, 480, 40, 630, 450, 330, 120,
    480, 350, 210, 70, 470, 370, 100, 20, 460, 450, 140, 0, 700, 80, 30, 40, 275, 255, 245, 190,
    330, 160, 130, 50, 630, 360, 180, 110, 490, 265, 170, 90, 440, 235, 120, 60, 560, 200, 340, 150,
    470, 360, 330, 60, 460, 255, 200, 70, 700, 350, 40, 170, 440, 340, 110, 50, 265, 190, 160, 130,
    450, 245, 370, 150, 480, 275, 100, 10, 630, 490, 80, 20, 560, 235, 90, 30, 210, 180, 120, 140,
    460, 210, 255, 245, 440, 120, 180, 160, 700, 370, 90, 10, 560, 150, 100, 20, 340, 200, 235, 140,
    450, 275, 190, 50, 630, 350, 130, 80, 480, 40, 110, 30, 490, 265, 330, 70, 470, 360, 170, 60
    ]

# baseline predictions to compare against
pr_preds = [
    491, 251, 150, 10, 350, 281, 233, 73, 458, 328, 263, 37, 536, 476, 67, 190, 599, 379, 80, 23,
    562, 199, 298, 46, 404, 369, 96, 222, 513, 115, 107, 52, 440, 318, 131, 30, 428, 183, 162, 59,
    462, 205, 302, 23, 366, 377, 136, 55, 563, 400, 147, 10, 412, 256, 179, 196, 491, 502, 43, 16, 
    247, 447, 270, 130, 537, 227, 49, 35, 430, 342, 71, 82, 586, 171, 313, 163, 284, 156, 234, 90,
    531, 329, 148, 22, 415, 354, 385, 32, 494, 178, 329, 203, 403, 287, 132, 51, 476, 508, 94, 32, 
    270, 247, 227, 60, 594, 259, 105, 10, 354, 227, 191, 78, 570, 183, 155, 68, 462, 169, 306, 105,
    484, 441, 124, 23, 503, 390, 103, 149, 420, 228, 293, 161, 529, 78, 64, 45, 329, 349, 360, 39, 
    578, 216, 15, 15, 638, 310, 243, 33, 205, 205, 254, 180, 552, 464, 268, 55, 234, 185, 143, 114,
    437, 222, 140, 68, 332, 383, 238, 15, 576, 290, 189, 204, 502, 320, 82, 128, 602, 307, 103, 74,
    544, 344, 245, 10, 483, 170, 89, 30, 408, 255, 157, 61, 523, 473, 162, 37, 458, 103, 195, 56,
    489, 296, 129, 21, 552, 415, 140, 36, 499, 250, 45, 10, 469, 309, 330, 60, 399, 285, 160, 112, 
    457, 243, 190, 183, 342, 517, 261, 205, 222, 99, 93, 106, 587, 372, 231, 87, 354, 168, 212, 72,
    623, 98, 117, 42, 395, 372, 117, 42, 323, 323, 195, 126, 481, 333, 79, 42, 495, 495, 155, 42,
    567, 414, 209, 254, 650, 225, 238, 201, 244, 175, 187, 164, 395, 350, 93, 51, 280, 434, 360, 155,
    521, 404, 135, 60, 374, 218, 255, 147, 643, 306, 92, 142, 436, 231, 218, 175, 551, 155, 22, 22,
    568, 424, 189, 70, 535, 162, 70, 10, 340, 289, 104, 98, 278, 354, 127, 197, 616, 255, 175, 78,
    471, 375, 143, 134, 601, 217, 203, 26, 537, 355, 162, 134, 521, 185, 79, 134, 553, 337, 170, 15,
    406, 265, 283, 86, 337, 295, 241, 100, 445, 431, 234, 15, 305, 118, 26, 91, 627, 179, 153, 52,
    395, 314, 226, 76, 464, 304, 337, 115, 384, 167, 144, 174, 439, 232, 244, 10, 542, 360, 258, 65,
    607, 360, 92, 65, 574, 405, 154, 127, 478, 213, 40, 76, 515, 288, 109, 98, 239, 132, 185, 22,
    418, 298, 179, 101, 600, 224, 130, 158, 430, 405, 211, 188, 370, 393, 122, 152, 75, 95, 10, 122,
    509, 309, 393, 206, 540, 239, 145, 86, 470, 232, 265, 138, 351, 335, 286, 172, 450, 319, 165, 122,
    540, 151, 175, 98, 610, 436, 175, 82, 499, 244, 24, 131, 421, 225, 92, 76, 328, 315, 294, 58, 
    589, 391, 51, 120, 452, 216, 108, 10, 350, 339, 267, 195, 471, 280, 304, 39, 513, 201, 126, 24,
    450, 171, 117, 55, 403, 387, 117, 78, 437, 494, 202, 15, 561, 93, 122, 59, 464, 338, 178, 216,
    350, 243, 226, 10, 605, 325, 165, 129, 299, 313, 185, 66, 376, 284, 140, 106, 543, 235, 254, 150,
    250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,
    250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,
    453, 228, 272, 158, 376, 216, 165, 103, 538, 290, 136, 88, 561, 177, 114, 95, 419, 235, 301, 142,
    432, 201, 193, 109, 512, 250, 122, 10, 394, 209, 65, 184, 471, 348, 331, 136, 407, 320, 171, 88
    ]

player_subs = {"P046": "P029", "P016": "P064", "P047": "P012", "P004": "P012", 
                  "P008": "P000", "P056": "P064", "P052": "P026", "P028": "P058", "P002": "P084",
                  "P063": "P033", "P030": "P035", "P009": "P053", "P097": "P136", "P099": "P136", "P096": "P136",
                  "P133": "P084", "P020": "P022", "P064": "P035", "P023": "P012", "P055": "P053", "P130": "P017",
                  "P038": "P064", "P125": "P051", "P144": "P036", "P050": "P061",
                  "P124": "P053", "P025": "P062", "P039": "P004", "P059": "P061", "P073": "P026",
                  "P015": "P062", "P137": "P026", "P065": "P053", "P027": "P017", "P040": "P044", "P075": "P097",
                  "P037": "P033", "P115": "P038", "P043": "P035", "P000": "P021", "P045": "P058", "P001": "P045",
                  "P042": "P049", "P057": "P062", "P029": "P029", "P062": "P038", "P035": "P084", "P067": "P067",
                  "P033": "P053", "P031": "P042", "P019": "P017", "P017": "P019", "P022": "P001", "P010": "P035",
                  "P036": "P055", "P060": "P057", "P071": "P019", "P136": "P099", "P127": "P099",
                  "P147": "P032", "P148": "P053", "P145": "P054", "P149": "P053", "P146": "P035", "P141": "P016",
                  "P150": "P030", "P143": "P010", "P155": "P027", "P151": "P027", "P152": "P096",
                  "P071": "P073", "P153": "P097", "P154": "P097", "P142": "P055", "P105": "P155", "P129": "P047",
                  "P139": "P097", "P007": "P153", "P003": "P154", "P078": "P054", "P156": "P027", "P079": "P041",
                  "P069": "P153", "P140": "P007", "P138": "P026", "P077": "P141", "P068": "P034", "P076": "P023",
                  "P006": "P035", "P202": "P026", "P203": "P038", "P178": "P029", "P204": "P056", "P179": "P056",
                  "P205": "P140", "P098": "P054", "P210": "P010", "P213": "P073", "P215": "P097", "P216": "P097",
                  "P218": "P097", "P217": "P096", "P219": "P032", "P220": "P011", "P221": "P016", "P222": "P059",
                  "P118": "P019", "P229": "P153", "P228": "P051", "P230": "P202", "P223": "P034", "P021": "P063",
                  "P225": "P097", "P231": "P063", "P227": "P067", "P105": "P141", "P224": "P062", "P226": "P027",
                  "P214": "P054", "P232": "P202", "P233": "P064", "P200": "P005", "P208": "P056",
                  "P072": "P141", "P186": "P038"}

# events whose impact on player stats is reduced
reduced_events = [22]

class C1_class:
    def __init__(self):
        self.pdict = fill_pdict()
        self.eve_str = compute_eve_str(self.pdict)
        self.e_dict = {"EVE_1": -11, "EVE_3": -10, "EVE_4": -9, "EVE_5": -8, "EVE_6": -7, "EVE_9": -6,
                       "EVE_11": -5, "EVE_13": -4, "EVE_14": -3, "EVE_15": -2, "EVE_18": -1,
                       "EVE_19": 0, "EVE_21": 1, "EVE_24": 2, "EVE_25": 3, "EVE_27": 4, "EVE_29": 5, 
                       "EVE_31": 6, "EVE_33": 7, "EVE_34": 8, "EVE_37": 9, "EVE_39": 10, "EVE_41": 11,
                       "EVE_42": 12, "EVE_43": 13, "CUR": 15}
    # returns the predicted result of a previous event, with option to override with new roster
    def prev_sim(self, event, roster=None):
        results = predict(self.pdict, self.e_dict[event], self.eve_str, roster=roster)
        return results
    # returns player ranking results
    def pr_results(self):
        pr = generate_pr(self.pdict, self.eve_str)
        return pr
    # returns predicted result of a new event, occuring at current time
    def cur_sim(self, roster):
        results = predict(self.pdict, self.e_dict["CUR"], self.eve_str, roster=roster)
        return results
    # computes and returns single event player scores
    def eve_pr(self, event):
        ind = min(max(3, self.e_dict[event]+6), self.e_dict["CUR"])
        weights = train(self.pdict, self.eve_str, ind)
        if(event in ["EVE_2", "EVE_7", "EVE_12", "EVE_17", "EVE_23", "EVE_32", "EVE_36", "EVE_44"]):
            # need the uncounted data for the above events
            dict_nc = fill_pdict(True, event)
            eval_pr = eve_pr_calc(dict_nc, -11, weights)
            return eval_pr
        
        return eve_pr_calc(self.pdict, self.e_dict[event], weights)
    
#########################################################################################################

old_pla = ["P018", "P086", "P066", "P131", "P132", "P118", "P078", "P127", "P129", "P134", "P109",
          "P100", "P106", "P126"]

# instantiate all_rosters by reading from file
mypath = os.path.dirname(__file__)
txtfile = os.path.join(mypath[:-2], "All_Teamsets.txt")
f = open(txtfile, 'r')
lines = f.read().split("\n")
teams = [x.split("|") for x in lines]
all_rosters = [[x.split(", ") for x in y] for y in teams]

all_rosters = [all_rosters[i] for i in event_select]

# player class storing stats events lists for each stat
class Player:
    def __init__(self, num_eve):
        self.time = [-1] * num_eve
        self.place = [-1] * num_eve
        self.num_last7 = [-1] * num_eve

class fullPlayer:
    def __init__(self):
        self.total = 0
        self.total2 = 0
        self.place = 0
        self.numtot = 0

def eve_pr_calc(pdict, eve, weights, d=default_d):

    loc = eve + 11
    matrix = []
    plist = []
    
    # create stats matrix of every player in the particular event
    for pla in pdict:
        if(pdict[pla].time[loc] == -1):
            continue
        z = pdict[pla]
        matrix.append([z.time[loc], max(z.time[loc], 0) * abs(z.time[loc]), z.place[loc], 1, 0])
        plist.append([pla])
    
    matrix = np.delete(matrix, d, axis=1)

    # compute scores using weights vector
    vals = np.matmul(matrix, weights)

    # formula for adjusting all event scores to be above 0, maintains relative differences
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


def build_player(pdict, loc, pla, eve_str, decay):
    inc = loc
    depth = 0
    div_tot = 0
    outlier1 = []
    outlier2 = []
    event_player = fullPlayer()

    while(inc >= 7):
        # increase decay for reduced_events
        revert = False
        if(inc+1 in reduced_events):
            depth += 5
            revert = True
        
        value = pdict[pla].time[inc]
        # if player was in this event (value is not -1)
        if value != -1:
            # adjustment for event comp, scaled to better reflect std
            value = value + (eve_str[inc] - eve_str[loc+1])/20
            event_player.total += value * pow(decay, depth)
            event_player.total2 += max(value, 0) * abs(value) * pow(decay, depth)
            div_tot += pow(decay, depth)
            # stored values to detect outliers
            if(depth < 6):
                outlier1.append(value * pow(decay, depth))
                outlier2.append(pow(decay, depth))
            # adjustment for event comp
            value = pdict[pla].place[inc] + eve_str[inc] - eve_str[loc+1]
            event_player.place += value * pow(decay, depth)
        
        # reduce event number, increase depth of decay
        inc -= 1
        depth += 1
        if(revert):
            depth -= 5
    
    ## for handling outliers
    # if div_tot != 0: 
    #     if len(outlier1) > 4:
    #         # reconstruct original values
    #         full_values = [outlier1[i] / outlier2[i] for i in range(len(outlier2))]
    #         aver = sum(full_values) / len(full_values)
    #         for k in range(len(full_values)):
    #             if(abs(full_values[k] - aver) > 0.05):
    #                 event_player.total -= outlier1[k]
    #                 event_player.total2 -= max(full_values[k], 0) * abs(full_values[k]) * pow(decay, depth)
    #                 div_tot -= outlier2[k]
    
    # determine weighted average for all values
    event_player.total /= div_tot
    event_player.total2 /= div_tot
    event_player.place /= div_tot
    return event_player

def compute_eve_pla(pdict, loc, pla, eve_str, decay, ispr):
    # if player not initialized (or haven't played many recent events and doing predictions)
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
    pla_a.total = (pla_a.total + pla_b.total) / 2
    pla_a.total2 = (pla_a.total2 + pla_b.total2) / 2
    pla_a.place = (pla_a.place + pla_b.place) / 2
    return pla_a
    
# read in the stats sheets and initialize a dictionary of stats
def fill_pdict(nc=False, event=None):
    pdict = {}

    # if not counted, read from uncounted times
    if(nc):
        txtfile = os.path.join(os.path.dirname(__file__),"times_uncounted.txt")
    else:
        txtfile = os.path.join(os.path.dirname(__file__),"times.txt")
    f = open(txtfile, "r")
    text = f.read()
    f.close()

    events = text.split("\n===\n")

    for e_num, eve_stats in enumerate(events):
        lines = eve_stats.split("\n")

        # for filling uncounted events, only read the selected event, and remove the heading
        if(nc):
            if(lines[0] != event):
                continue
            lines = lines[1:]
            # set e_num back to 0, only filling one entry for each stat
            e_num = 0

        totlist = []
        for line in lines:
            # read line, which begins with the player name
            vals = line.split("\t")
            pla = vals[0]

            if(pla not in pdict):
                # if not counted, initialize player with 1 entry per stat, else number of events
                pdict[pla] = Player(1 if nc else len(events))

            # append to totlist a tuple of the player and their time
            totlist.append([pla, float(vals[1])])

        # sort the overall list by time to get placements
        totlist = sorted(totlist, key=lambda x: (x[1]))
        for i in range(len(totlist)):
            # save each placement
            pdict[totlist[i][0]].place[e_num] = max(40 - i, 1)
        
        # for unbalanced events, override the mean and std with precomputed values
        overrides = {17: [499.9, 30.2806], 18: [403.5216, 22.1289], 19: [496.23, 25.2112], 21: [505.74, 23.66], 22: [473.41, 28.17]}
        overrides_nc = {"EVE_32": [499.9, 30.2806], "EVE_2": [416.8, 58.65], "EVE_12": [385.46, 43.4221]}

        # check need for overrides for counted and non-counted events
        if(nc and event in overrides_nc):
            aver = overrides_nc[event][0]
            std = overrides_nc[event][1]
        elif(not nc and e_num in overrides):
            aver = overrides[e_num][0]
            std = overrides[e_num][1]
        else:
            # compute the mean and std for all but the last 5 player times
            templist = sorted([x[1] for x in totlist])[0:-5]
            aver = np.average(templist)
            std = np.std(templist)
        
        for i in range(len(totlist)):
            # compute normalized times and save to the stat
            pla = totlist[i][0]
            # invert so larger values are "better", intuitive
            x = round(-1 * (totlist[i][1] - aver)  / std, 4)
            pdict[pla].time[e_num] = max(x, -3.5)

    
    # if non-counted, return generated pdict as is
    if(nc):
        return pdict

    for pla in pdict:
        # determine number of previous events played in the last 7 for every event (plus current status)
        pdict[pla].num_last7.append(-1)
        for i in range(8, len(pdict[pla].num_last7)):
            eve_tot = 0
            # only start counting events at event 7 (beginning of modern results)
            for k in range(max(7, i-7), i):
                    if(pdict[pla].place[k] != -1):
                        eve_tot += 1
            # for early events, manually set to 2 if played in 1
            if(i < 9 and eve_tot == 1):
                eve_tot = 2
            pdict[pla].num_last7[i] = eve_tot

    ### 
    # Prepare and save a stats sheet for Exterior Stats
    string = ""
    for i in range(len(pdict["P053"].time)):
        for pla in pdict:
            if(pdict[pla].time[i] != -1):
                string += (pla + "\t" + str(pdict[pla].time[i]) + "\n")
        string = string[:-1] + "\n===\n"
    string = string[:-5]

    mypath = os.path.dirname(__file__)[:-2] + "Exterior Stats"
    txtfile = os.path.join(mypath, "C1stats.txt")
    f = open(txtfile, 'w')
    f.write(string)
    f.close()
    ###

    return pdict

# iterates though events, finding average "place" metric going into each event
# effectively an event strength measure 
def compute_eve_str(pdict, decay=0.9):
    eve_str = [0] * 100
    new_eve_str = []
    for i in range(len(pdict["P053"].place)):
        new_eve_str.append(0)
        # only compute strength for events 9 and higher
        if(i >= 9):
            roster = all_rosters[i-9]
            # sum place average going into the event for all players
            for team in roster:
                for pla in team:
                    eve_pla = compute_eve_pla(pdict, i-1, pla, eve_str, decay, False)
                    new_eve_str[-1] += eve_pla.place
    
    # find average over all events for normalization
    aver = np.mean(new_eve_str[9:])

    # normalize and scale all strengths
    for i in range(9, len(new_eve_str)):
        new_eve_str[i] -= aver
        new_eve_str[i] /= 40

    # set final (current) event to 0, neutral strength
    new_eve_str.append(0)

    return new_eve_str


def train(pdict, eve_str, eve_num=None, decay=0.9, d=default_d):
    loc = 8
    matrix = []

    # when eve_num not supplyed, train using all data
    if(eve_num is None):
        eve_num = len(all_rosters) - 2

    for event in all_rosters[:2+eve_num]:
        average = 0
        # for every player on every team, initialize a matrix row containing their stat averages before the event
        for team in event:
            for pla in team:
                z = compute_eve_pla(pdict, loc, pla, eve_str, decay, False)
                matrix.append([z.total, z.total2, z.place, 1])
                # event strength measure, average time
                average += z.total
        average = average / 40
        # append average time of the whole event as another trainable variable
        for j in range(40):
            matrix[-1 - j].append(average)
        loc += 1
    
    # feature selection, delete stats numbered by d
    matrix = np.delete(matrix, d, axis=1)

    # solve linear regression problem using matrix algebra
    tmatrix = np.transpose(matrix)
    fmatrix = np.matmul(np.linalg.inv(np.matmul(tmatrix, matrix)), tmatrix)
    # weights for each stat in the linear computation
    weights = np.matmul(fmatrix, results[0:80+eve_num*40])

    return weights

def predict(pdict, eve_num, eve_str, decay=0.9, d=default_d, roster=None):
    # get weights by training with all data prior to event
    weights = train(pdict, eve_str, eve_num, decay, d)

    # if roster is supplied, override event #eve_num with roster
    if(roster is not None):
        predict_roster = roster
    else:
        predict_roster = all_rosters[eve_num+2]
    
    input_matrix = []
    average = 0

    # compute every player's stats going into the event
    for team in predict_roster:
        for pla in team:
            z = compute_eve_pla(pdict, eve_num+10, pla, eve_str, decay, False)
            input_matrix.append([z.total, z.total2, z.place, 1])
                # event strength measure, average time
            average += z.total
    average = average / 40

    for j in range(40):
        input_matrix[j].append(average)
    
    # feature selection, delete stats numbered by d
    input_matrix = np.delete(input_matrix, d, axis=1)
    
    result = np.matmul(input_matrix, weights)
    
    x = min(result)
    # scoring shouldn't be negative, manually adjust scores up while maintaining order
    if(x < 10):
        m = 20 / (30 - x)
        b = 30 - 30*m
        for i in range(len(result)):
            if(result[i] < 30):
                result[i] = result[i]*m + b
                
    # normalize and round all scores
    norm = 10040 / sum(result)
    result = result * norm
    for j in range(40):
        result[j] = float(round(result[j]))
    result = result.astype(int)

    return result

def generate_pr(pdict, eve_str, decay=0.9, d=default_d):
    weights = train(pdict, eve_str, decay=decay, d=d)

    matrix_pr = []
    pr = []

    # calculate stats for every player, then apply regression derived formulas
    average = 0
    for pla in pdict:
        if(pla in old_pla):
            continue
        z = compute_eve_pla(pdict, len(all_rosters)+8, pla, eve_str, decay, True)
        matrix_pr.append([z.total, z.total2, z.place, 1])
        average += z.total
        pr.append([pla])
    average /= len(pdict)
    for row in matrix_pr:
        row.append(average)
    
    matrix_pr = np.delete(matrix_pr, d, axis=1)
    
    scores = np.matmul(matrix_pr, weights)

    for i in range(len(pr)):
        pr[i].append(float(round(scores[i])))

    # sort player score pairs in descending order
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
    
    # to prevent negative values, apply transform to lower values
    x = min([x[1] for x in pr])
    if(x < 0):
        m = 30 / (30 - x)
        b = 30 - 30*m
        for i in range(len(pr)):
            if(pr[i] < 30):
                pr[i] = pr[i]*m + b

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

    # initialize C1 class
    cla = C1_class()

    # all events leaderboard code
    perfList = []
    for key in cla.e_dict:
        if(key != "CUR"):
            val = cla.eve_pr(key)
            for value in val:
                perfList.append([value[1], value[0] + key[-2:]])
    perfList = sorted(perfList, reverse=True)

    # eve_pr test
    val = cla.eve_pr("EVE_37")

    # if current event, return player rankings
    if(cla.e_dict["CUR"] == eve_num):
        pr = generate_pr(cla.pdict, cla.eve_str, decay, default_d)
        string = ""
        for player in pr:
            string = string + (player[0] + "\t" + str(player[1]) + "\n")
        
        check = 0
    else:
        scores = predict(cla.pdict, eve_num, cla.eve_str, decay, default_d)

        # check against true results for mse accuracy
        results_test = results[80+eve_num*40:120+eve_num*40]
        error = 0
        errorm = []
        for j in range(40):
            error += np.power((scores[j] - results_test[j]), 2)
            errorm.append(scores[j] - results_test[j])
        error = error / 40
        error = np.sqrt(error)
        
        # evaluate mse accuracy of another prediction method for comparison
        error2 = 0
        error2m = []
        pr_preds_range = pr_preds[0+eve_num*40:40+eve_num*40]
        for j in range(40):
            error2 += np.power((pr_preds_range[j] - results_test[j]), 2)
            error2m.append(pr_preds_range[j] - results_test[j])
        error2 = error2 / 40
        error2 = np.sqrt(error2)

        # evaluate mse accuracy of naive prediction method for comparison
        error3 = 0
        equal_values = [250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,
        250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250]
        for j in range(40):
            error3 += np.power((equal_values[j] - results_test[j]), 2)
        error3 = error3 / 40
        error3 = np.sqrt(error3)

        print(str(error) + " " + str(error2))
        check = 1