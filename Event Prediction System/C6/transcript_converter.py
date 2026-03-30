import os.path


def find_str(eve_num):
    # for first event, strength is 1
    if(eve_num == 0):
        return [1] * 10
    roster = all_rosters[eve_num]
    strengths = []
    for team in roster:
        total = 0
        for pla in team:
            num_events = 0
            stat1 = 0
            for i in range(eve_num+1):
                mult = 1
                # increase weight of current event greatly
                if(i == eve_num):
                    mult = 1.375
                if(pdict[pla][i] != -1):
                    # add decayed stat1 amt to weighted sum
                    stat1 += pdict[pla][i] * mult * pow(0.95, eve_num-i)
                    num_events += mult * pow(0.95, eve_num-i)

            # compute average (time decayed) in stat1 for the player
            stat1 = stat1 / num_events
            total += stat1 / 11
        # append full team's result
        strengths.append(total)

    return strengths

event_select = [10,12,13,14,15,16,19,20,21,23,24,25,27,29,31,32,33,34,35,36,38,39,41,42]

# instantiate all_rosters by reading from file
mypath = os.path.dirname(__file__)
txtfile = os.path.join(mypath[:-2], "All_Teamsets.txt")
f = open(txtfile, 'r')
lines = f.read().split("\n")
teams = [x.split("|") for x in lines]
all_rosters = [[x.split(", ") for x in y] for y in teams]

all_rosters = [all_rosters[i] for i in event_select]

team_dict = {"vpode": 0, "njogb": 1, "qimpfcn": 2, "nvfjoae": 3, "beinpa": 4, "nioaeti": 5, "bsro": 6, "mgieroq": 7, "bosrj": 8, "eopjtp": 9}

mypath = os.path.dirname(__file__)
txtfile = os.path.join(mypath,"g6_transcripts.txt")
f = open(txtfile, "r")
text = f.read()

new_str = ""

events = text.split("\n===\n")
for enum, event in enumerate(events):
    pla_dict = {}
    rounds = event.split("\n---\n")
    # initialize data structure for all players in event
    tnums = {}
    for i, team in enumerate(all_rosters[enum]):
        for pla in team:
            pla_dict[pla] = [0,0,0,0,0,[set(),set(),set()]]
            tnums[pla] = i

    for i in range(3):
        # store when downs occur, and when final eliminations occur
        stat3_dict = {}
        elim_list = []

        lines = rounds[i].split("\n")

        # get useful stat from last line summary, store for each player (slot 4)
        stat6 = lines[-1].split(" ")
        for j in range(10):
            for k in range(4):
                pla_dict[all_rosters[enum][j][k]][4] += int(stat6[4 + 3*j])
        
        # remove summary lines
        lines = lines[:-2]

        for line in lines:
            # remove trailing spaces and split
            words = line.rstrip().split(" ")
            if "nbifo pwkf" in line:
                # note downed player, attacker, and time
                downed = words[1]
                attacker = words[5]
                time = 60 * int(words[0][1]) + int(words[0][3:5])
                stat3_dict[downed] = (time, attacker)
                # add to "downs" stat for the attacker
                pla_dict[attacker][1] += 1

                # record each player interaction with the other's team
                pla_dict[downed][5][i].add(tnums[attacker])
                pla_dict[attacker][5][i].add(tnums[downed])
            if "gedfv kod wpev" in line or "orto kbro tof" in line:
                # revivals, remove player from downed dict and add revival
                pla_dict[words[1]][3] += 1
                del stat3_dict[words[1]]
            if(any([term in line for term in ["oerfg", "perjr", "fpxmfq", "ioe dowa otkr", "vjdkzo"]])):
                # converted elim
                if(words[1] in stat3_dict):
                    # if player was downed, elim time is when downed + credit attacker with elim
                    elimed = words[1]
                    attacker = stat3_dict[words[1]][1]
                    elim_list.append([stat3_dict[elimed][0], elimed])
                    pla_dict[attacker][0] += 1
                    del stat3_dict[elimed]
                else:
                    # otherwise, no attacker, elim time is current time
                    time = int(words[0][1]) * 60 + int(words[0][3:5])
                    elim_list.append([time, words[1]])
            if "eliminated" in line:
                # all downed members of team eliminated must be processed
                team_num = team_dict[words[1]]
                for pla in all_rosters[enum][team_num]:
                    if(pla in stat3_dict):
                        # elim time is when downed + credit attacker with elim
                        elim_list.append([stat3_dict[pla][0], pla])
                        pla_dict[stat3_dict[pla][1]][0] += 1
                        del stat3_dict[pla]

        # sort elim list by times (first value of sublists)
        elim_list = sorted(elim_list)
        all_plas = set(pla_dict.keys())
        # add player's placement for the round
        for j in range(len(elim_list)):
            pla = elim_list[j][1]
            pla_dict[pla][2] += j+1
            all_plas.remove(pla)
        
        # players who don't appear survived, so add 40 (max placement)
        for pla in all_plas:
            pla_dict[pla][2] += 40

    # compute average over 3 rounds for placement
    for pla in pla_dict:
        pla_dict[pla][2] = round(float(pla_dict[pla][2] / 3), 2)

    ### team interaction logic
    for i in range(3):
        for team in all_rosters[enum]:
            # Requirement: both 1st+4th and 2nd+3rd pairs have interacted with other team
            # if so, keep that team
            full = list((pla_dict[team[0]][5][i] | pla_dict[team[3]][5][i]) & (pla_dict[team[1]][5][i] | pla_dict[team[2]][5][i]))
            for pla in team:
                pla_dict[pla][5][i] = full

    # combination of team interactions from all 3 rounds
    for pla in pla_dict:
        pla_dict[pla][5] = pla_dict[pla][5][0] + pla_dict[pla][5][1] + pla_dict[pla][5][2]

    # convert stats for all players into formatted string
    for pla in pla_dict:
        p = pla_dict[pla]
        new_str += "%s\t%d\t%d\t%.2f\t%d\t%d\t" % (pla, p[0], p[1], p[2], p[3], p[4])
        for num in p[5]:
            new_str += "%d," % (num)
        new_str = new_str[:-1]
        new_str += "\n"
    new_str += "===\n"

new_str = new_str[:-5]
pdict = {}
events = new_str.split("\n===\n")
# initialize elim totals for each player in each event
for i in range(len(events)):
    lines = events[i].split("\n")
    for line in lines:
        words = line.split("\t")
        pla = words[0]
        # if player has not appeared, initialize null values (-1) for all past events
        if(pla not in pdict):
            pdict[pla] = [-1] * i
        pdict[pla].append(int(words[1]))
    for pla in pdict:
        # if player did not appear in this event, add null value (-1)
        if(len(pdict[pla]) != i + 1):
            pdict[pla].append(-1)
val_list = []
full_str = ""
for j in range(len(events)):
    strengths = find_str(j)
    lines = events[j].split("\n")
    first = lines[0].split("\t")
    # if comp hasn't been initialized, initialize
    if(len(first) != 8):
        for i in range(int(len(lines) / 4)):
            words = lines[i*4].split("\t")
            # get overall strength faced using team's faced list
            faced_str = 0
            vals = words[6].split(",")
            for val in vals:
                faced_str += strengths[int(val)]
            faced_str /= len(vals)
            # append opp. strength to all 4 players' stats
            for k in range(4):
                full_str += lines[i*4+k] + "\t" + str(round(faced_str, 2)) + "\n"
            val_list.append(val)
        full_str += "===\n"
    # otherwise just reconstruct result
    else:
        full_str += events[j] + "\n===\n"

# remove final redundant seperator
full_str = full_str[:-5]

check = 0