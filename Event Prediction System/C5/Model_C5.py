import numpy as np
import os.path

##############################################################################################

# Consistently Updated Globals and Classes

default_decay = 0.8
default_d = [1, 5]

eve_num = 17

# numbered events where this contest appeared
event_select = [5, 7, 9, 10, 13, 14, 15, 19, 22, 23, 24, 26, 28, 30, 31, 32, 36, 38, 42, 43]

results = [340, 224, 248, 264, 132, 104, 236, 232, 188, 56, 352, 280, 184, 520, 168, 442, 176, 12, 204, 76,
    376, 240, 394, 204, 616, 372, 28, 120, 292, 300, 236, 504, 212, 374, 176, 168, 386, 60, 72, 480,
    240, 244, 378, 156, 232, 232, 506, 52, 430, 280, 320, 112, 284, 308, 108, 84, 488, 160, 596, 128,
    384, 172, 280, 216, 160, 128, 492, 208, 24, 72, 152, 136, 368, 398, 316, 44, 260, 260, 252, 300,
    478, 290, 132, 364, 428, 104, 128, 148, 448, 168, 116, 216, 450, 128, 52, 368, 348, 442, 340, 76,
    442, 360, 120, 284, 212, 84, 140, 328, 224, 326, 208, 168, 296, 96, 140, 160, 400, 208, 340, 200,
    332, 284, 128, 144, 340, 184, 342, 28, 656, 292, 180, 72, 232, 312, 180, 224, 220, 152, 36, 252,
    394, 498, 244, 350, 348, 176, 36, 120, 112, 116, 88, 408, 512, 248, 200, 220, 348, 360, 340, 252,
    248, 132, 164, 184, 612, 256, 364, 236, 180, 192, 160, 216, 248, 376, 204, 380, 220, 528, 160, 100,
    216, 320, 12, 374, 256, 248, 244, 120, 390, 196, 180, 100, 342, 188, 308, 358, 350, 236, 132, 256,
    296, 104, 276, 92, 156, 344, 220, 386, 308, 448, 32, 56, 490, 304, 420, 76, 278, 448, 100, 148, 
    512, 132, 232, 384, 404, 200, 232, 156, 340, 48, 160, 204, 252, 272, 44, 336, 240, 418, 168, 244,
    112, 228, 148, 136, 112, 124, 404, 184, 378, 196, 132, 320, 216, 244, 690, 92, 204, 352, 474, 160,
    300, 332, 284, 112, 184, 388, 356, 188, 384, 244, 104, 148, 300, 328, 116, 278, 108, 520, 424, 56,
    264, 236, 180, 44, 488, 256, 176, 56, 204, 386, 120, 164, 220, 488, 228, 232, 468, 402, 164, 196,
    144, 156, 438, 328, 420, 192, 112, 44, 328, 276, 92, 220, 622, 232, 316, 204, 108, 324, 200, 232,
    558, 512, 208, 84, 180, 192, 382, 88, 454, 216, 220, 272, 388, 180, 260, 60, 372, 536, 80, 168,
    284, 184, 116, 116, 290, 264, 172, 140, 124, 240, 224, 120, 426, 388, 328, 96, 428, 252, 296, 88,
    450, 470, 152, 172, 232, 224, 224, 232, 436, 92, 296, 200, 208, 132, 120, 132, 430, 228, 454, 116, 
    364, 308, 144, 116, 268, 424, 312, 284, 342, 268, 124, 416, 422, 236, 160, 112, 168, 260, 92, 140,
    606, 296, 186, 176, 422, 240, 288, 156, 342, 92, 64, 184, 610, 196, 264, 224, 312, 200, 352, 84,
    280, 312, 184, 172, 324, 220, 180, 128, 368, 224, 342, 148, 488, 288, 336, 40, 140, 232, 132, 28,
    268, 268, 164, 88, 358, 430, 224, 288, 312, 372, 152, 256, 398, 148, 116, 112, 356, 268, 260, 248,
    546, 288, 156, 36, 558, 180, 356, 72, 504, 180, 72, 4, 312, 224, 184, 80, 176, 296, 466, 184,
    624, 284, 328, 72, 324, 386, 264, 292, 422, 280, 208, 164, 372, 456, 140, 136, 458, 352, 72, 52,
    500, 184, 260, 140, 478, 240, 148, 128, 392, 220, 276, 88, 288, 300, 228, 196, 40, 60, 60, 48,
    88, 180, 196, 124, 396, 300, 236, 176, 280, 296, 284, 192, 224, 500, 88, 116, 304, 490, 200, 96,
    546, 452, 296, 144, 488, 256, 144, 180, 220, 380, 132, 144, 324, 248, 212, 184, 476, 112, 204, 104,
    268, 344, 208, 148, 252, 164, 184, 188, 490, 216, 220, 164, 418, 220, 156, 248, 648, 60, 104, 56,
    372, 506, 268, 260, 248, 312, 76, 32, 264, 268, 200, 148, 332, 308, 240, 394, 528, 276, 104, 68,
    428, 288, 172, 48, 188, 116, 220, 64, 628, 500, 252, 64, 636, 180, 184, 108, 464, 418, 252, 148,
    296, 152, 196, 168, 384, 284, 312, 160, 284, 364, 256, 84, 200, 184, 100, 358, 400, 264, 352, 68,
    368, 450, 240, 152, 280, 404, 68, 24, 396, 240, 212, 156, 364, 208, 284, 196, 284, 116, 96, 164,
    132, 378, 420, 272, 394, 184, 180, 60, 272, 212, 116, 60, 454, 370, 380, 200, 594, 284, 240, 56,
    304, 208, 176, 108, 446, 324, 244, 112, 656, 292, 152, 48, 252, 320, 320, 140, 563, 414, 128, 12,
    368, 356, 318, 88, 397, 228, 64, 80, 240, 272, 132, 208, 300, 216, 360, 152, 390, 68, 180, 328,
    287, 294, 222, 236, 318, 350, 13, 209, 227, 177, 157, 120, 436, 65, 87, 135, 320, 168, 277, 220,
    473, 258, 227, 58, 595, 390, 285, 153, 478, 228, 81, 63, 229, 244, 121, 128, 327, 128, 253, 189,
    377, 275, 145, 147, 464, 273, 103, 76, 261, 274, 118, 63, 255, 201, 362, 91, 484, 186, 275, 117,
    243, 272, 217, 77, 294, 223, 108, 189, 310, 295, 137, 54, 338, 423, 115, 127, 535, 258, 166, 72
    ]

pr_preds = [249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249,
    249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249,
    337, 87, 244, 392, 253, 119, 352, 201, 394, 87, 43, 294, 296, 169, 220, 386, 194, 333, 270, 103,
    270, 487, 103, 251, 393, 218, 210, 54, 340, 299, 220, 134, 431, 199, 160, 332, 243, 270, 354, 218,
    376, 154, 109, 79, 249, 12, 238, 217, 471, 290, 272, 63, 503, 303, 258, 251, 257, 479, 74, 439, 
    386, 178, 196, 229, 316, 360, 225, 77, 449, 65, 95, 344, 372, 89, 111, 275, 275, 272, 238, 241,
    370, 299, 240, 225, 418, 147, 324, 62, 365, 269, 107, 126, 234, 278, 521, 171, 271, 344, 243, 237,
    206, 350, 294, 203, 195, 426, 212, 324, 115, 433, 157, 74, 349, 334, 187, 232, 45, 376, 308, 98,
    342, 169, 257, 94, 424, 358, 57, 99, 396, 398, 213, 84, 352, 323, 137, 219, 529, 452, 241, 225,
    267, 350, 299, 234, 414, 235, 128, 67, 342, 278, 44, 152, 350, 247, 250, 128, 235, 127, 269, 175,
    460, 369, 209, 51, 317, 333, 326, 61, 512, 107, 341, 168, 392, 324, 182, 72, 374, 334, 115, 61,
    272, 241, 241, 61, 373, 132, 238, 99, 241, 277, 142, 185, 577, 300, 324, 81, 504, 197, 223, 142,
    353, 312, 200, 98, 424, 414, 272, 180, 380, 196, 245, 310, 304, 154, 66, 43, 371, 371, 344, 154, 
    519, 243, 154, 44, 448, 255, 261, 58, 196, 196, 241, 240, 500, 350, 211, 46, 220, 194, 216, 176,
    556, 437, 156, 221, 395, 311, 283, 201, 468, 194, 123, 118, 551, 258, 176, 127, 294, 226, 209, 83,
    411, 107, 84, 177, 418, 244, 189, 102, 389, 290, 340, 148, 416, 227, 314, 55, 71, 282, 156, 156,
    249, 425, 277, 172, 314, 334, 303, 156, 427, 342, 66, 111, 403, 197, 63, 150, 290, 342, 170, 240,
    395, 239, 149, 67, 299, 297, 194, 77, 564, 341, 101, 40, 451, 127, 158, 168, 339, 402, 272, 257,
    454, 377, 190, 76, 198, 345, 187, 123, 434, 375, 163, 62, 496, 377, 217, 174, 620, 319, 49, 123,
    469, 327, 263, 74, 605, 305, 120, 61, 273, 263, 213, 107, 443, 299, 172, 73, 165, 124, 123, 123,
    249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,
    249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,
    572, 349, 129, 175, 311, 188, 327, 118, 650, 255, 81, 140, 399, 307, 72, 136, 512, 163, 80, 100,
    446, 421, 143, 271, 468, 224, 129, 63, 271, 272, 130, 130, 251, 378, 167, 350, 376, 228, 143, 39,
    269, 363, 113, 138, 358, 163, 301, 72, 499, 291, 109, 33, 645, 331, 52, 51, 431, 361, 194, 33,
    382, 414, 199, 103, 330, 508, 251, 121, 378, 426, 229, 43, 314, 141, 51, 270, 643, 116, 200, 33,
    314, 359, 110, 38, 314, 325, 96, 38, 341, 287, 202, 123, 315, 400, 240, 212, 357, 328, 240, 181,
    415, 429, 270, 141, 288, 404, 110, 96, 270, 275, 235, 38, 357, 334, 266, 110, 478, 287, 217, 122,
    248, 280, 232, 182, 420, 348, 186, 83, 412, 370, 187, 32, 282, 357, 294, 83, 548, 207, 186, 83,
    285, 309, 229, 104, 470, 271, 286, 95, 417, 216, 94, 244, 375, 240, 186, 161, 309, 287, 174, 186,
    227, 130, 266, 135, 349, 244, 189, 163, 462, 210, 178, 75, 382, 160, 283, 30, 392, 271, 286, 224,
    490, 227, 254, 117, 484, 425, 175, 38, 609, 318, 12, 108, 327, 346, 300, 75, 389, 379, 175, 55,
    249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,
    249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,
    ]

player_subs = {"P115": "P019", "P106": "P019", "P059": "P061", "P036": "P061", "P058": "P051", "P024": "P033",
                  "P010": "P051", "P018": "P022", "P023": "P019", "P055": "P000", "P064": "P051", "P042": "P042",
                  "P041": "P022", "P060": "P000", "P054": "P000", "P049": "P061", "P004": "P019", "P131": "P019",
                  "P014": "P049", "P144": "P054", "P017": "P051", "P038": "P051", "P135": "P019",
                  "P047": "P033", "P099": "P127", "P096": "P127", "P097": "P127", "P067": "P000", "P016": "P051",
                  "P026": "P019", "P056": "P051", "P006": "P051", "P034": "P022", "P046": "P061", 
                  "P043": "P051", "P126": "P127", "P008": "P019", "P021": "P033", "P136": "P127", "P012": "P000", 
                  "P052": "P019", "P028": "P033", "P005": "P051", "P002": "P062", "P063": "P033", "P125": "P019",
                  "P020": "P042", "P050": "P051", "P124": "P036", "P009": "P064", "P030": "P019",
                  "P039": "P035", "P073": "P006", "P015": "P064", "P137": "P125", "P065": "P059",
                  "P040": "P059", "P027": "P026", "P075": "P060", "P134": "P033", "P086": "P000", "P120": "P064",
                  "P121": "P026", "P076": "P097", "P077": "P099", "P068": "P020", "P069": "P097", "P071": "P096",
                  "P070": "P012", "P072": "P011", "P141": "P027", "P033": "P030", "P142": "P065", "P118": "P063",
                  "P105": "P027", "P129": "P023", "P150": "P006", "P007": "P069", "P139": "P077", "P079": "P097",
                  "P003": "P077", "P146": "P012", "P156": "P056", "P037": "P055", "P138": "P071", "P140": "P016",
                  "P143": "P096", "P098": "P011", "P179": "P016", "P178": "P005", "P180": "P057", 
                  "P181": "P064", "P182": "P015", "P183": "P056", "P184": "P020", "P185": "P012",
                  "P186": "P071", "P187": "P022", "P188": "P032", "P189": "P096", "P190": "P011", "P193": "P034",
                  "P194": "P016", "P192": "P047", "P191": "P035", "P195": "P001", "P196": "P062", "P207": "P025",
                  "P208": "P143", "P209": "P069", "P210": "P073", "P206": "P030", "P211": "P002", 
                  "P204": "P016", "P212": "P097", "P213": "P005", "P214": "P011", "P202": "P136", "P223": "P036",
                  "P224": "P060", "P225": "P077", "P105": "P019", "P226": "P030", "P227": "P054", "P220": "P057",
                  "P219": "P011", "P149": "P009", "P078": "P005", "P231": "P060", "P232": "P027", "P200": "P059", 
                  "P208": "P016", "P203": "P056", "P186": "P019", "P233": "P009", "P079": "P030",
                  "P061": "P012"
                  }

class C5_class:
    def __init__(self):
        self.pdict = fill_pdict()
        self.eve_str = compute_eve_str(self.pdict)
        self.e_dict = {"EVE_3": -5, "EVE_5": -4, "EVE_6": -3, "EVE_8": -2, "EVE_10": -1, "EVE_11": 0,
                       "EVE_14": 1, "EVE_15": 2, "EVE_16": 3, "EVE_21": 4, "EVE_24": 5, "EVE_25": 6, "EVE_26": 7,
                       "EVE_28": 8, "EVE_30": 9, "EVE_32": 10, "EVE_33": 11, "EVE_34": 12, "EVE_38": 13, 
                       "EVE_44": 15, "EVE_45": 16, "CUR": 17}
        self.nc_eves = {"EVE_1": 4, "EVE_2": 4, "EVE_12": 5, "EVE_17": 6, "EVE_40": 15}
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
        if(event in self.nc_eves):
            weights = train(self.pdict, self.eve_str, self.nc_eves[event], d=[4,5])
            # need the uncounted data for the above events
            dict_nc = fill_pdict(True, event)
            return eve_pr_calc(dict_nc, -5, weights,d=[4,5])
        ind = min(max(5, self.e_dict[event]+3), self.e_dict["CUR"])
        weights = train(self.pdict, self.eve_str, ind, d=[4,5])
        return eve_pr_calc(self.pdict, self.e_dict[event], weights, d=[4,5])

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
        self.num = [-1] * num_eve
        self.time = [-1] * num_eve
        self.rounds = [[-1,-1,-1] for _ in range(num_eve)]
        self.t8 = [-1] * num_eve
        self.t4 = [-1] * num_eve
        self.scores = [-1] * num_eve
        self.num_last7 = [-1] * num_eve

class eventPlayer:
    def __init__(self):
        self.time = 0
        self.time3 = 0
        self.num = 0
        self.num2 = 0
        self.t8 = 0
        self.t4 = 0
        self.scores = 0

def eve_pr_calc(pdict, eve, weights, d=default_d):
    loc = eve + 5
    matrix = []
    plist = []

    # create stats matrix of every player in the particular event
    for pla in pdict:
        if(pdict[pla].num[loc] == -1):
            continue
        z = pdict[pla]
        matrix.append([z.num[loc], pow(z.num[loc], 2), z.time[loc], pow(z.time[loc], 2), z.t8[loc], z.t4[loc], z.scores[loc], 1, 20])
        plist.append([pla])

    matrix = np.delete(matrix, d, axis=1)

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

    vals = [int((val-250)*1.25+250) for val in vals]

    for i in range(len(plist)):
        plist[i].append(float(round(vals[i])))

    plist = (sorted(plist,key=lambda x: (x[1])))
    plist.reverse()
    return plist

# computes time decayed averages for all stats up to time loc
# performs outlier analysis
def build_player(pdict, loc, pla, eve_str, decay):
    inc = loc
    depth = 0
    div = 0
    outlier1 = [[],[],[],[],[]]
    outlier2 = []
    event_player = eventPlayer()

    while(inc >= 0):
        value = pdict[pla].num[inc]

        # if stats for event don't exist (== -1) skip
        if value != -1:
            # adjustment for event comp
            value *= eve_str[inc] / eve_str[loc+1]

            event_player.num += value * pow(decay, depth)
            event_player.num2 += pow(value,2) * pow(decay, depth)
            div += pow(decay,depth) 
            if(depth < 8):
                outlier1[0].append(value * pow(decay, depth))
                outlier2.append(pow(decay, depth))

            value = pdict[pla].time[inc] * eve_str[loc+1] / eve_str[inc]
            event_player.time += value * pow(decay, depth)
            event_player.time3 += pow(value,3) * pow(decay, depth)
            if(depth < 8):
                outlier1[1].append(value * pow(decay, depth))

            value = pdict[pla].t8[inc] * eve_str[inc] / eve_str[loc+1]
            event_player.t8 += value * pow(decay, depth)
            if(depth < 8):
                outlier1[2].append(value * pow(decay, depth))

            value = pdict[pla].t4[inc] * eve_str[inc] / eve_str[loc+1]
            event_player.t4 += value * pow(decay, depth)
            outlier1[3].append(value * pow(decay, depth))

            value = pdict[pla].scores[inc] * eve_str[inc] / eve_str[loc+1]
            event_player.scores += value * pow(decay, depth)
            if(depth < 8):
                outlier1[4].append(value * pow(decay, depth))
       
       # increase time decay (more if player played event)
        inc -= 1
        if(pdict[pla].num[inc] != -1):
            depth += 1
        else:
            depth += 0.5
        # large gap present, increase decay even more
        if(inc in [13]):
            depth += 1

    # do outlier analysis if more than 3 events
    if(len(outlier2) > 3):
        full_values = [outlier1[1][i] / outlier2[i] for i in range(len(outlier2))]
        mean = np.mean(full_values)
        std = np.std(full_values)

        for k in range(len(full_values)):
            # if num is outlier 
            if(abs((full_values[k] - mean)/std) > 1.5):
                j = 0
                for i, var in enumerate(vars(event_player)):
                    # reduce outlier's effect by 50%, for squared or cubed values need to compute first
                    if(var == "num2"):
                        setattr(event_player, var, vars(event_player)[var] - pow(outlier1[0][k], 2) * 0.5 / outlier2[k])
                        j += 1
                    elif(var == "time3"):
                        setattr(event_player, var, vars(event_player)[var] - pow(outlier1[1][k], 3) * 0.5 / pow(outlier2[k], 2))
                        j += 1
                    else:
                        setattr(event_player, var, vars(event_player)[var] - outlier1[i-j][k] * 0.5)
                div -= outlier2[k] * 0.5               
    
    # divide all stat totals by divisor to determine weighted average        
    for var in vars(event_player):
        setattr(event_player, var, vars(event_player)[var] / div)
    
    return event_player

# logic for established versus new players, calls build_player() to get stats (including subs)
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

def fill_pdict(nc=False, event=None):
    pdict = {}
    mypath = os.path.dirname(__file__)

    # select uncounted or counted events
    if(nc):
        txtfile = os.path.join(mypath,"places_uncounted.txt")
    else:
        txtfile = os.path.join(mypath,"places.txt")       
    f = open(txtfile, "r")
    text = f.read()
    f.close()
    
    # read in all stats
    events = text.split("\n===\n")
    for enum, eve_stats in enumerate(events):
        rounds = eve_stats.split("\n---\n")
        # first "round" is scores
        scores = rounds[0]
        rounds = rounds[1:]

        lines = scores.split("\n")

        # for non-counted events, only read in the event with the relevant header
        if(nc):
            if(lines[0] != event):
                continue
            lines = lines[1:]
            # set enum back to 0, only filling one entry for each stat
            enum = 0

        for line in lines:
            vals = line.split("\t")
            pla = vals[0]
            if(pla not in pdict):
                # if not counted, initialize player with 1 entry per stat, else number of events
                pdict[pla] = Player(1 if nc else len(events))
            
            pdict[pla].scores[enum] = int(vals[1])

        # read in results for all 3 rounds
        for j in range(len(rounds)):
            best_pla = ""
            aver = 0
            maxi = 0
            lines = rounds[j].split("\n")
            temp_data = []
            for line in lines:
                values = line.split(" ")
                del values[0:2]
                # read in the data for the 4 players on each team
                for i in range(4):
                    pla = values[i*3]
                    pdict[pla].rounds[enum][j] = 41 - float(values[i*3+1][2:-1])
                    if(values[i*3+1][2:-1] == "1"):
                        best_pla = pla
                    if(j == 0):
                        pdict[pla].time[enum] = 0
                        pdict[pla].t8[enum] = 0
                        pdict[pla].t4[enum] = 0
                    if(j == 2):
                        pdict[pla].num[enum] = sum(pdict[pla].rounds[enum]) / 3
                        pdict[pla].rounds[enum].sort()

                    # temporarily store the time so it can be normalized using the entire round's data
                    time = float(values[i*3+2][0]) * 60 + float(values[i*3+2][2:4])
                    temp_data.append([pla, time])
                    aver += time
                    if(maxi < time): maxi = time
            
            # compute the mean and std of the roster for this round
            aver /= 40
            std = 0
            for data in temp_data:
                std += pow(data[1] - aver, 2)
                # add additional time to winner
                if(data[0] == best_pla):
                    data[1] += 15
            std = pow(std/40, 1/2)

            # get the 8th and 4th best times
            ordered_data = sorted(temp_data, key=lambda x: x[1], reverse=True)
            eighth = ordered_data[7][1]
            fourth = ordered_data[3][1]
            
            for data in temp_data:
                # normalize player time and add to Player instance
                pdict[data[0]].time[enum] += (data[1] - aver) / (3 * std)
                # generate features, square of time outlasted above 8th and 4th place players
                pdict[data[0]].t8[enum] += pow(max(data[1] - eighth, 0), 2) / 100
                pdict[data[0]].t4[enum] += pow(max(data[1] - fourth, 0), 2) / 100

    if(nc):
        return pdict
    
    for pla in pdict:
        # determine number of previous events played in the last 7 for every event (plus current status)
        pdict[pla].num_last7.append(-1)
        for i in range(1, len(pdict[pla].num_last7)):
            eve_tot = 0
            # check last 7 events, or until start of event history
            for k in range(max(0, i-7), i):
                    if(pdict[pla].num[k] != -1):
                        eve_tot += 1
            # for early events, manually set to 2 if played in 1
            if(i < 3 and eve_tot == 1):
                eve_tot = 2
            pdict[pla].num_last7[i] = eve_tot
                    
    return pdict

# iterates though events, finding average "place" metric going into each event
# effectively an event strength measure 
def compute_eve_str(pdict, decay=0.9):
    eve_str = [1] * 100
    new_eve_str = []
    for i in range(len(pdict["P053"].num)):
        new_eve_str.append(1)
        # only compute strength for events 9 and higher
        if(i >= 2):
            new_eve_str[-1] -= 1
            roster = all_rosters[i-2]
            # sum place average going into the event for all players
            for team in roster:
                for pla in team:
                    eve_pla = compute_eve_pla(pdict, i-1, pla, eve_str, decay, False)
                    new_eve_str[-1] += eve_pla.num
    
    # find average over all events for normalization
    aver = np.mean(new_eve_str[2:])

    # normalize and scale all strengths
    for i in range(2, len(new_eve_str)):
        new_eve_str[i] /= aver
        new_eve_str[i] = round(new_eve_str[i], 4)

    # set final (current) event to 0, neutral strength
    new_eve_str.append(1)

    return new_eve_str

def train(pdict, eve_str, eve_num=None, decay=default_decay, d=default_d):
    loc = 1
    matrix = []

    if(eve_num is None):
        eve_num = len(all_rosters) - 3
    
    # initialize stat priors for every team in every event
    for event in all_rosters[:3+eve_num]:
        average = 0
        for team in event:
            for pla in team:
                # generate stat input rows for each player
                z = compute_eve_pla(pdict, loc, pla, eve_str, decay, False)
                matrix.append([z.num, z.num2, z.time, z.time3,
                z.t8, z.t4, z.scores, 1])
                average += z.num
        average = average / 40
        for j in range(40):
            matrix[-1 - j].append(average)
        loc += 1

    # select specific stats for training
    matrix = np.delete(matrix, d, axis=1)

    # solving for the weight vector
    matrix1 = matrix[:120+eve_num*40]
    tmatrix1 = np.transpose(matrix1)
    fmatrix1 = np.matmul(np.linalg.inv(np.matmul(tmatrix1, matrix1)), tmatrix1)
    weights = np.matmul(fmatrix1, results[0:120+eve_num*40])

    return weights

def predict(pdict, eve_num, eve_str, decay=default_decay, d=default_d, roster=None):
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
        for pla in team:
            z = compute_eve_pla(pdict, eve_num+4, pla, eve_str, decay, False)
            pdict_eve[pla] = z
            input_matrix.append([z.num, z.num2, z.time, z.time3, z.t8, z.t4, z.scores, 1])
            average += z.num

    # append average time of the whole event as another trainable variable
    average /= 40
    for j in range(40):
        input_matrix[j].append(average)
    
    # feature selection, delete stats numbered by d
    input_matrix = np.delete(input_matrix, d, axis=1)

    result = np.matmul(input_matrix, weights)

    norm = 9960 / sum(result)
    result = result * norm

    # manually increase std for more realistic predictions
    for j in range(40):
        result[j] = round((result[j] - 250) * 1.1 + 250)
    result = result.astype(int)
        
    return result

def generate_pr(pdict, eve_str, decay=default_decay, d=default_d):
    weights = train(pdict, eve_str, decay=decay, d=d)

    matrix_pr = []
    pr = []

    # compute stats for every player to identify their individual skill level
    average = 0
    for pla in pdict:
        z = compute_eve_pla(pdict, len(all_rosters)+1, pla, eve_str, decay, True)
        matrix_pr.append([z.num, z.num2, z.time3, z.time, z.t8, z.t4, z.scores, 1])
        average += z.num
        pr.append([pla])
    average = average / len(pdict)
    for row in matrix_pr:
        row.append(average)

    # using previously derived weight vector compute scores for every player
    matrix_pr = np.delete(matrix_pr, d, axis=1)
    scores = np.matmul(matrix_pr, weights)

    for i in range(len(pr)):
        pr[i].append(float(round(scores[i])))

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

    avg_scores = 0
    total = 0
    for player in pr:
        avg_scores += player[1] * float(pfreq_dict[player[0]])
        total += float(pfreq_dict[player[0]])
    avg_scores /= total
    # normalize to 250, then increase std for more reasonable player scores
    for player in pr:
        player[1] = round(player[1] * (250 / avg_scores))
        player[1] = round((player[1]-250) * 1.1 + 250)
    
    return pr

if __name__ == "__main__":

    cla = C5_class()
    pdict = cla.pdict

    perfList = []
    for key in cla.e_dict:
        if(key != "CUR"):
            val = cla.eve_pr(key)
            for value in val:
                perfList.append([value[1], value[0] + key[-2:]])
    perfList = sorted(perfList, reverse=True)

    # eve_pr test
    val = cla.eve_pr("EVE_33")

    # if current event, return player rankings
    if(cla.e_dict["CUR"] == eve_num):
        pr = generate_pr(pdict, cla.eve_str, default_decay, default_d)
        string = ""
        for player in pr:
            string = string + (player[0] + "\t" + str(player[1]) + "\n")
        
        check = 0
    else:
        scores = predict(pdict, eve_num, cla.eve_str, default_decay, default_d)

        # for simulated events, check against true results for mse accuracy
        results_test = results[120+eve_num*40:160+eve_num*40]
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
        equal_values = [249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,
        249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,249]
        for j in range(40):
            error3 += np.power((equal_values[j] - results_test[j]), 2)
        error3 = error3 / 40
        error3 = np.sqrt(error3)

        errom = [int(x) for x in errorm]
        errom2 = [pow(x,2) for x in errom]
        error2m2 = [pow(x,2) for x in error2m]

        print(error)
        check = 0