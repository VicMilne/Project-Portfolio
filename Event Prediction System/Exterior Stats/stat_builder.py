import numpy as np
import os.path

###
# for processing substitution stats in team based Gamemodes
# code computes score for contest x prior to every instance of contest y being played
# stores these results in text files in the contest's respective folder
###

full = ["EVE_1", "EVE_2", "EVE_3", "EVE_4", "EVE_5", "EVE_6", "EVE_7", "EVE_8", "EVE_9", "EVE_10", "EVE_11", "EVE_12", "EVE_13",
        "EVE_14", "EVE_15", "EVE_16", "EVE_18", "EVE_19", "EVE_20", "EVE_21", "EVE_22", "EVE_23", "EVE_24", "EVE_25", "EVE_26",
        "EVE_27", "EVE_28", "EVE_29", "EVE_30", "EVE_31", "EVE_32", "EVE_33", "EVE_34", "EVE_35", "EVE_36", "EVE_37", "EVE_38", 
        "EVE_39", "EVE_40", "EVE_41", "EVE_42", "EVE_43", "EVE_44", "EVE_45"]
m4_list = ["EVE_4", "EVE_5", "EVE_6", "EVE_7", "EVE_8", "EVE_9", "EVE_10", "EVE_12", "EVE_13", "EVE_15", "EVE_16", "EVE_18",
           "EVE_19", "EVE_21", "EVE_22", "EVE_24", "EVE_25", "EVE_26", "EVE_27", "EVE_29", "EVE_31", "EVE_33", "EVE_34", "EVE_36",
           "EVE_39", "EVE_41", "EVE_43", "EVE_44"]
m3_list = ["EVE_1", "EVE_2", "EVE_4", "EVE_5", "EVE_6", "EVE_7", "EVE_8", "EVE_9", "EVE_12", "EVE_14", "EVE_20", "EVE_23",
           "EVE_28", "EVE_31", "EVE_36", "EVE_37", "EVE_39", "EVE_40", "EVE_42", "EVE_45"]
m6_list = ["EVE_30", "EVE_32", "EVE_35", "EVE_38", "EVE_39", "EVE_41", "EVE_43", "EVE_45"]
m5_list = ["EVE_11", "EVE_13", "EVE_14", "EVE_15", "EVE_16", "EVE_18", "EVE_21", "EVE_22", "EVE_23", "EVE_25", "EVE_26", "EVE_27",
           "EVE_29", "EVE_31", "EVE_33", "EVE_34", "EVE_35", "EVE_36", "EVE_37", "EVE_38", "EVE_40", "EVE_41", "EVE_43", "EVE_44",
           "EVE_45"]

eve_list_m1 = ["EVE_1", "EVE_3", "EVE_4", "EVE_5", "EVE_6", "EVE_9", "EVE_11", "EVE_13", "EVE_14", "EVE_15", "EVE_18",
               "EVE_19", "EVE_21", "EVE_24", "EVE_25", "EVE_27", "EVE_29", "EVE_31", "EVE_33", "EVE_34", "EVE_37",
               "EVE_39", "EVE_41", "EVE_42", "EVE_43", "EVE_44"]

eve_list_m2 = ["EVE_1", "EVE_3", "EVE_6", "EVE_8", "EVE_10", "EVE_11", "EVE_14", "EVE_15", "EVE_16", "EVE_18", "EVE_19", 
               "EVE_21", "EVE_24", "EVE_25", "EVE_28", "EVE_29", "EVE_30", "EVE_35", "EVE_37", "EVE_39", "EVE_42", "EVE_43"]

eve_list_m8 = ["EVE_3", "EVE_4", "EVE_5", "EVE_6", "EVE_7", "EVE_8", "EVE_9", "EVE_10", "EVE_11", "EVE_14", "EVE_18", 
               "EVE_19", "EVE_20", "EVE_21", "EVE_22", "EVE_24", "EVE_25", "EVE_28", "EVE_29", "EVE_30", "EVE_35", "EVE_37", "EVE_40",
               "EVE_42", "EVE_43", "EVE_44", "EVE_45"]

eve_list_m7 = ["EVE_1", "EVE_3", "EVE_4", "EVE_7", "EVE_9", "EVE_10", "EVE_11", "EVE_13", "EVE_15", "EVE_19", "EVE_22",
               "EVE_26", "EVE_28", "EVE_31", "EVE_37", "EVE_39", "EVE_43"]

# External Sourced Stats
eve_list_m9 = ["EVE_1", "EVE_3", "EVE_4", "EVE_5", "EVE_8", "EVE_9", "EVE_10", "EVE_13", "EVE_14", "EVE_16", "EVE_18", 
               "EVE_19", "EVE_21", "EVE_24", "EVE_26", "EVE_27", "EVE_30", "EVE_34", "EVE_35", "EVE_36", "EVE_37", "EVE_42", "EVE_44",
               "EVE_45"]
eve_list_mA = ["EVE_13", "EVE_20", "EVE_23", "EVE_26", "EVE_28", "EVE_29", "EVE_30", "EVE_31", "EVE_35", "EVE_40", "EVE_42", "EVE_44"]

# compute time decayed average going into each event in contest_list
def compute_local(pdict, contest_list):
    # overwrite last event in list with most recent event, maintain most up to date results
    contest_list[-1] = full[-1]
    condict = {}
    decay = 1.1
    for num in contest_list:
        ind = full.index(num)
        # for each player, compute time decayed average from "ind" event backwards
        # remains at default zero if no data is available
        for pla in pdict:
            tot = 0
            div = 0
            for i in range(ind, -1, -1):
                tot += pdict[pla][i] * pow(decay, i)
                if(pdict[pla][i] != 0): div += pow(decay, i)
            # light normalization factor, catches players with limited data
            tot /= (div + 0.01)
            # add result to list in condict
            if(pla not in condict):
                condict[pla] = [tot]
            else:
                condict[pla].append(tot)

    # extract data from condict and format for text file
    string = ""
    for pla in condict:
        string += pla + " "
        act = False
        for i in range(len(condict[pla])):
            if(act or condict[pla][i] != 0):
                string += str(round(condict[pla][i], 3)) + " "
                act = True
            else:
                string += "100 "
        string = string[:-1] + "\n"

    string = string[:-1]

    return string

def export_contest(contest, eve_list, cols):
    # read contest data from stats file
    f = open("{}stats.txt".format(contest), "r")
    text = f.read()
    f.close()

    eves = text.split("\n===\n")

    pdict = {}
    contest_num = 0
    # for each event, use the relevant columns to compute the contest score and save in pdict
    # if a player is not in an event, or the event doesn't contain the contest, use 0
    for event_text in eves:
        # get the overall index of the current event
        overall_num = full.index(eve_list[contest_num])
        lines = event_text.split("\n")
        stored_list = []
        for line in lines:
            vals = line.split("\t")
            pla = vals[0]
            if(pla not in pdict):
                pdict[pla] = [0] * overall_num
            # extend the current list with 0s up to the current event
            if(len(pdict[pla]) != overall_num):
                pdict[pla].extend([0] * (overall_num - len(pdict[pla])))
            score = 0

            # if contest is C7, vals[2] may be null value, if so replace with vals[1]
            if(contest == "C7" and float(vals[2]) == -1):
                vals[2] = vals[1]

            # compute the contest score (linear combination of the columns)
            for i in range(len(vals)):
                if(i in cols):
                    score += float(vals[i]) * cols[i]
            
            pdict[pla].append(score)
            stored_list.append(score)
            # if contest is C8, handle boundary cases
            if(contest == "C8" and contest_num < 4 and vals[1] == "1"):
                pdict[pla][-1] /= 2
                stored_list[-1] /= 2
            elif(contest == "C8" and pla == "P098" and contest_num == 20):
                pdict[pla][-1] = pdict["P011"][-1]
        
        # if contest is C1, do not normalize as the results are already normalized in a specific way
        if(contest != "C1"):

            # compute average and std
            aver = sum(stored_list) / len(stored_list)
            std = np.std(stored_list)

            # if player list is full length (most recent event included a player result), normalize
            for pla in pdict:
                if(len(pdict[pla]) == overall_num + 1):
                    pdict[pla][overall_num] = (pdict[pla][overall_num] - aver) / std

        contest_num += 1
    
    # extend every player list to most recent event with 0s
    overall_num = len(full)
    for pla in pdict:
        if(len(pdict[pla]) != overall_num):
            pdict[pla].extend([0] * (overall_num - len(pdict[pla])))

    # compute time decayed contest scores for every event in every relevant contest
    # save results to the respective files

    string = compute_local(pdict, m4_list)

    mypath = os.path.dirname(os.path.dirname(__file__))
    f = open(os.path.join(mypath, "C4\\C4_{}.txt".format(contest)), "w")
    f.write(string)
    f.close()

    string = compute_local(pdict, m3_list)

    mypath = os.path.dirname(os.path.dirname(__file__))
    f = open(os.path.join(mypath, "C3\\C3_{}.txt".format(contest)), "w")
    f.write(string)
    f.close()

    string = compute_local(pdict, m6_list)

    mypath = os.path.dirname(os.path.dirname(__file__))
    f = open(os.path.join(mypath, "C6\\C6_{}.txt".format(contest)), "w")
    f.write(string)
    f.close()

    string = compute_local(pdict, m5_list)

    mypath = os.path.dirname(os.path.dirname(__file__))
    f = open(os.path.join(mypath, "C6\\C6_{}.txt".format(contest)), "w")
    f.write(string)
    f.close()

# export the contest results for all 6 substitution contests
# dict selects which columns in the data files to use, with what weights
if __name__ == '__main__':
    export_contest("C2", eve_list_m2, {1: 1, 2: 1})

    export_contest("C8", eve_list_m8, {2: 1, 3: 1, 4: -0.25, 5: -0.25})

    export_contest("C7", eve_list_m7, {1: 1, 2: 1})

    export_contest("C9", eve_list_m9, {1: 1})

    export_contest("C1", eve_list_m1, {1: 1})

    export_contest("CA", eve_list_mA, {2: 1})