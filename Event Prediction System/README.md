# Linear Regression Event Prediction System

## Overview

This project concerns a niche periodic event series where a large, mostly static player base competes in a selection of different competitive contests (drawn from a set of 8) to earn the most points. The events are team-based, with 10 teams of 4 competing each event. Teams are not consistent event to event, as players are consistently grouped in different combinations.

The goal of this project is to predict the performance of each team in an upcoming event using each player's past stats. Linear Regression is used to learn from the limited data available.

Most identifiable references to the event, including its name, the names of the players, and many variable names and text data have been anonymized. The functionality is preserved, but some code may be harder to follow due to generically renamed variables and keywords in comments.

## Standard Methodology

The 8 contest types that the event cycles through are distinct and employ slightly different skillsets and parameters for success, so each contest has its own unique linear regression model. However, the standard approach for each contest is fairly similar.
- The best numerical weights for solving all past instances of the problem ("given a player/team's statistical profile prior to an event, what score will they achieve?") are computed using linear regression.
- The "input" values are typically time-decayed averages of a player's stats measured from past performances in the contest.
- The "output" values are player or team scores in the contest during a particular event.
- Input stats are often normalized by event and/or adjusted based on the evaluated skill of an event's roster.
- Many models employ outlier analysis to remove outlier single-event performances.
- As a slow stream of new players are added to the roster every event, their statistical priors are manually set to established players with similar skill levels.

## Contest Overviews

Below is a brief description of each contest to appear in the event and an overview of how its characteristics inform the design of its corresponding linear regression model.

### Contest 1

This contest features a race to complete several objectives. Each player on a team works separately and never interacts, so scoring and predictions are done on an individual basis. The tracked stats for this contest are an individual's placement in the race and their time to complete it.
### Contest 2

This contest is mostly team-based and requires teams to compete against each other to achieve a limited set of objectives. While some objectives are achieved as a team, others can be attributed to particular individuals. The tracked stats are each player's number of completions of each type of objective, as well as the team's overall totals and each individual's percentage of the total.

### Contest 3
This contest is entirely team-based and contains no statistics that can be attributed to individuals, as players play distinct roles that interact to earn points. Two methods are used to make predictions despite this hurdle.
- Naive player scores are obtained by "solving" for each player's skill using linear regression. In the Ax = b equation, the b vector contains team scores over multiple events, the x vector is the size of the player base and represents each individual's score, and A is a sparse matrix where each row represents a team and each column represents a player (A[i][j] = 1 -> team i contains player j). Solving for x computes an estimate for each player's average contribution to all their teams' scores over the set of events.
- Individual player stats are imported from other contests. While these stats are not directly applicable to contest 3, overlapping skill sets mean they have some utility and provide additional trainable data.

### Contest 4
Another entirely team-based contest, distinct from M3 but requiring a similar approach for predictions.

### Contest 5
This contest features a "last person standing" competition over 3 rounds. Similar to contest 1, each player's performance is entirely independent of their team, so the training and prediction targets are individual scores. The tracked stats for this contest are a player's placement and their time lasted for each round.

### Contest 6
Similar to contest 2, this contest involves teams competing for a shared pool of objectives, with some objectives shared as a team and others attributed to specific individuals. Along with the tracked stats from the contest, contest 6 also trains on stats imported from other contests, in the same way that contests 3 and 4 do.

### Contest 7
Contest 7 features single team vs. team matches, where one player from each team is selected as the "attacker" while the other 3 act as "defenders". Each defender is tasked with defending an objective, while attackers attempt to capture them. As this structure is baked into the contest and its scoring, the model estimates which players will attack and defend, and computes the team's pre-event stats (average attack time, defend time, etc) using these estimates. The model also attempts to adjust for attacker and defender strength when processing stats.

### Contest 8
This contest asks each team to score points while avoiding elimination. The relevant stats trained on include the various sources of points and the overall time lasted before elimination. Using the same methodology as contests 3 and 4, each individual player's "contribution" to their time lasted is estimated and used as another input stat.