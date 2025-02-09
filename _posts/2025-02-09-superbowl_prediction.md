---
layout: post
title: Predicting Super Bowl
date: 2025-01-04
categories: [Football, NFL]
tags: [Data Science, Supervised Learning, NFL, Football]
render_with_liquid: false
---

I have yet to encounter anyone who wants the Chiefs to win the Super bowl. Unfortunately, the Chiefs are the favorites based on the sportsbooks. Fortunately, I was able to find some statistical models that (while less accurate than the sportsbooks) tell a different story. 

Each statistical model will have a brief overview with links for more detailed explanations. I will then go through how effective each model was at predicting the winners of NFL games, and finally give a prediction for the super bowl based on these models. 
## Spread

The spread is the difference that the sportsbooks think the favorite will win by. For instance, if the favorite is Kansas City and the spread is 6 points, that means there is a 50 percent chance Kansas City will win by more than 6 points and a 50 percent chance the Chiefs will win by less than 6 points.

## Simple Rating System

The simple rating system (SRS) is a rating system used by ProFootballReference. It determines a team's rating by adding together the team's average margin of victory plus their strength of schedule. Here is an in-depth reference from PFR on how these values are calculated. SRS can be interpreted as how many points better one team is than another. For instance, if the Eagles has an SRS rating of 10 and the Chiefs an SRS of 7, then SRS would predict that the Eagles would win by 3 if the teams played each other. In this article, I build on SRS to account for some weaknesses of this system. 

### Score Transform

One downside of the SRS is individual large wins can highly influence it. To diminish the impact of individual large wins, I applied a score transform shown below to the MoV of each game. This transformation reduces the impact of large wins. 

![Image](/assets/img/log_function.png)

### Exponential Decay

The SRS also treats all games as equally important (whether its the first game of the season or the most recent game). However, teams can both improve and get worse over the course of the season. I applied an exponential decay that makes games further in the past less important in the calculation. This decay makes each game worth 90 percent the weight of the game after it (so the most recent game has a weight of 1, 2 games ago a weight of 0.9, three games ago a weight of 0.81 etc.)

## Regression Ratings

Another approach to determine ratings is to use a linear model. To do this, I create a linear model where the input variables are indicators for each team and the output is the margin of victory for each game. I then fit this model to get ratings for each team that, like SRS, can be interpreted as how many points better one team is than another.

## Ratings for this Season

Below are the ratings for this season using different ratings systems. Two things worth noting:
1. Using the SRS + log transform results in KC making it in the top 5 because the log transform causes ratings to be influenced less by large wins. KC had a lot of close games this season, and thus didn't have the big wins to achieve a high SRS even though they only lost 2 games.
2. With the SRS + Decay model, teams that surged at the end of the season have a higher rating than without a decay because the decay causes the final games of the season to matter more than the games at the beginning of the season. This can be seen with LAR who is in the top ten for the SRS + Decay model because a strong finish to their season.

<ul><li style="color: #2e8b57"><span style="color: black">Top 5</span></li><li style="color: #98fb98"><span style="color: black">5-10</span></li><li style="color: #fb9898"><span style="color: black">Bottom 5-10</span></li><li style="color: #8a2e2e"><span style="color: black">Bottom 5</span></li></ul><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SRS</th>
      <th>SRS + Decay</th>
      <th>SRS + Log</th>
      <th>Regression</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CAR</th>
      <td style="background-color:#8a2e2e">-11.16</td>
      <td style="background-color:#8a2e2e">-2.93</td>
      <td style="background-color:#8a2e2e">-1.49</td>
      <td style="background-color:#8a2e2e">-10.48</td>
    </tr>
    <tr>
      <th>CLE</th>
      <td style="background-color:#8a2e2e">-9.30</td>
      <td style="background-color:#8a2e2e">-3.50</td>
      <td style="background-color:#8a2e2e">-1.53</td>
      <td style="background-color:#8a2e2e">-8.68</td>
    </tr>
    <tr>
      <th>TEN</th>
      <td style="background-color:#8a2e2e">-8.26</td>
      <td style="background-color:#8a2e2e">-3.45</td>
      <td style="background-color:#8a2e2e">-1.65</td>
      <td style="background-color:#8a2e2e">-7.91</td>
    </tr>
    <tr>
      <th>NYG</th>
      <td style="background-color:#8a2e2e">-8.08</td>
      <td style="background-color:#8a2e2e">-2.75</td>
      <td style="background-color:#fb9898">-1.25</td>
      <td style="background-color:#8a2e2e">-7.45</td>
    </tr>
    <tr>
      <th>NE</th>
      <td style="background-color:#8a2e2e">-7.73</td>
      <td style="background-color:#fb9898">-2.34</td>
      <td style="background-color:#8a2e2e">-1.39</td>
      <td style="background-color:#8a2e2e">-7.44</td>
    </tr>
    <tr>
      <th>JAX</th>
      <td style="background-color:#fb9898">-7.34</td>
      <td style="background-color:#8a2e2e">-2.63</td>
      <td style="background-color:#8a2e2e">-1.28</td>
      <td style="background-color:#fb9898">-6.96</td>
    </tr>
    <tr>
      <th>LVR</th>
      <td style="background-color:#fb9898">-6.58</td>
      <td style="background-color:#fb9898">-2.04</td>
      <td style="background-color:#fb9898">-1.18</td>
      <td style="background-color:#fb9898">-6.25</td>
    </tr>
    <tr>
      <th>DAL</th>
      <td style="background-color:#fb9898">-6.37</td>
      <td style="background-color:#fb9898">-1.97</td>
      <td>-0.59</td>
      <td style="background-color:#fb9898">-5.82</td>
    </tr>
    <tr>
      <th>NO</th>
      <td style="background-color:#fb9898">-4.25</td>
      <td style="background-color:#fb9898">-2.35</td>
      <td style="background-color:#fb9898">-0.74</td>
      <td style="background-color:#fb9898">-3.90</td>
    </tr>
    <tr>
      <th>NYJ</th>
      <td style="background-color:#fb9898">-4.01</td>
      <td>-1.52</td>
      <td style="background-color:#fb9898">-0.86</td>
      <td style="background-color:#fb9898">-3.91</td>
    </tr>
    <tr>
      <th>IND</th>
      <td>-3.57</td>
      <td style="background-color:#fb9898">-2.01</td>
      <td>-0.60</td>
      <td>-3.46</td>
    </tr>
    <tr>
      <th>MIA</th>
      <td>-2.59</td>
      <td>-0.31</td>
      <td>-0.48</td>
      <td>-2.52</td>
    </tr>
    <tr>
      <th>ATL</th>
      <td>-2.43</td>
      <td>-1.14</td>
      <td>-0.28</td>
      <td>-2.20</td>
    </tr>
    <tr>
      <th>CHI</th>
      <td>-2.40</td>
      <td>-1.42</td>
      <td style="background-color:#fb9898">-0.61</td>
      <td>-2.48</td>
    </tr>
    <tr>
      <th>SF</th>
      <td>-1.02</td>
      <td>-1.25</td>
      <td>-0.29</td>
      <td>-1.19</td>
    </tr>
    <tr>
      <th>HOU</th>
      <td>0.37</td>
      <td>0.00</td>
      <td>0.16</td>
      <td>0.38</td>
    </tr>
    <tr>
      <th>CIN</th>
      <td>1.34</td>
      <td>0.96</td>
      <td>0.45</td>
      <td>1.45</td>
    </tr>
    <tr>
      <th>SEA</th>
      <td>1.36</td>
      <td>0.55</td>
      <td>0.35</td>
      <td>1.07</td>
    </tr>
    <tr>
      <th>LAR</th>
      <td>1.47</td>
      <td style="background-color:#98fb98">1.33</td>
      <td>0.33</td>
      <td>1.12</td>
    </tr>
    <tr>
      <th>PIT</th>
      <td>1.76</td>
      <td>-0.11</td>
      <td>0.39</td>
      <td>1.80</td>
    </tr>
    <tr>
      <th>ARI</th>
      <td>2.19</td>
      <td>0.66</td>
      <td>0.19</td>
      <td>1.87</td>
    </tr>
    <tr>
      <th>LAC</th>
      <td>3.88</td>
      <td>1.21</td>
      <td style="background-color:#98fb98">0.73</td>
      <td>3.73</td>
    </tr>
    <tr>
      <th>WAS</th>
      <td style="background-color:#98fb98">4.00</td>
      <td style="background-color:#98fb98">1.48</td>
      <td style="background-color:#98fb98">0.76</td>
      <td style="background-color:#98fb98">4.08</td>
    </tr>
    <tr>
      <th>KC</th>
      <td style="background-color:#98fb98">4.40</td>
      <td style="background-color:#98fb98">1.25</td>
      <td style="background-color:#2e8b57">1.20</td>
      <td style="background-color:#98fb98">4.19</td>
    </tr>
    <tr>
      <th>MIN</th>
      <td style="background-color:#98fb98">5.08</td>
      <td>0.75</td>
      <td style="background-color:#98fb98">0.91</td>
      <td style="background-color:#98fb98">4.57</td>
    </tr>
    <tr>
      <th>DEN</th>
      <td style="background-color:#98fb98">5.13</td>
      <td style="background-color:#98fb98">1.95</td>
      <td>0.69</td>
      <td style="background-color:#98fb98">4.89</td>
    </tr>
    <tr>
      <th>TB</th>
      <td style="background-color:#98fb98">5.74</td>
      <td style="background-color:#98fb98">2.00</td>
      <td style="background-color:#98fb98">0.74</td>
      <td style="background-color:#98fb98">5.64</td>
    </tr>
    <tr>
      <th>GB</th>
      <td style="background-color:#2e8b57">7.65</td>
      <td style="background-color:#2e8b57">2.54</td>
      <td style="background-color:#98fb98">1.02</td>
      <td style="background-color:#2e8b57">7.05</td>
    </tr>
    <tr>
      <th>BUF</th>
      <td style="background-color:#2e8b57">9.14</td>
      <td style="background-color:#2e8b57">3.68</td>
      <td style="background-color:#2e8b57">1.37</td>
      <td style="background-color:#2e8b57">8.63</td>
    </tr>
    <tr>
      <th>PHI</th>
      <td style="background-color:#2e8b57">9.17</td>
      <td style="background-color:#2e8b57">5.26</td>
      <td style="background-color:#2e8b57">1.66</td>
      <td style="background-color:#2e8b57">9.05</td>
    </tr>
    <tr>
      <th>BAL</th>
      <td style="background-color:#2e8b57">10.01</td>
      <td style="background-color:#2e8b57">4.59</td>
      <td style="background-color:#2e8b57">1.54</td>
      <td style="background-color:#2e8b57">9.58</td>
    </tr>
    <tr>
      <th>DET</th>
      <td style="background-color:#2e8b57">12.40</td>
      <td style="background-color:#2e8b57">3.53</td>
      <td style="background-color:#2e8b57">1.74</td>
      <td style="background-color:#2e8b57">11.56</td>
    </tr>
  </tbody>
</table>

# Probability Forecasting

## Isotonic Regression
We now want to convert the difference in ratings between teams (or the spread), into a probability. To do this, we will use an approach called isotonic regression. The only assumption this isotonic regression makes is that, as the input variable increases, so does the output variable. This makes intuitive sense. The probability of a team winning when they are the ten point favorite should be higher than the probability of them winning when they are a 3 point favorite. Similarly, if a team's rating is 10 points higher than their opponents, then the probability of them winning should be greater than if the rating difference was only 5 points. Below is the graph of spread to probability that I used. 

![Image](/assets/img/isotonic.png)

## Gradient Boosting

To generalize isotonic regression from 1 input variable to multiple input variables, I used a statistical model called Gradient Boosting which is able to make the output isotonic in each input variable. Using this model, I can use both the spread and ratings to predict the probability of a team winning. Below is the output of this model when using the spread and the rating difference as features in the model

Looking at the top right and bottom left corners, we can see how the model incorporates both spread and rating difference information to increase its certainty when both models are very confident. 

![Image](/assets/img/2d-heatmap.png)

# Results 

Below are the results for 5 models:
- Spread: Isotonic regression fit on the game spread
- Spread + SRS: Gradient boosting fit on spread and SRS model using log transform and exponential decay
- SRS + Decay: Isotonic regression fit on SRS model using log transform and exponential decay
- Regression: Linear regression
- SRS: Simple rating system

We evaluate the results using two metrics
- Accuracy: How often did the team the model predicted would win (had >50 percent chance of winning) actually win
- Brier Score: Metric that accounts for certainty of prediction to evaluate how good predictions are. 
### Past 10 years

Below are the results average over all games after week 5 for each of the past 10 seasons.

<ul><li style="color: #C9B037"><span style="color: black">Best</span></li><li style="color: #D7D7D7"><span style="color: black">Second</span></li><li style="color: #AD8A56"><span style="color: black">Third</span></li></ul><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>accuracy</th>
      <th>brier score loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Spread</th>
      <td style="background-color:#D7D7D7">0.6752</td>
      <td style="background-color:#C9B037">0.2083</td>
    </tr>
    <tr>
      <th>Spread + SRS</th>
      <td style="background-color:#C9B037">0.6767</td>
      <td style="background-color:#D7D7D7">0.2093</td>
    </tr>
    <tr>
      <th>SRS + Decay</th>
      <td>0.6430</td>
      <td style="background-color:#AD8A56">0.2222</td>
    </tr>
    <tr>
      <th>Regression</th>
      <td style="background-color:#AD8A56">0.6435</td>
      <td>0.2224</td>
    </tr>
    <tr>
      <th>SRS</th>
      <td>0.6400</td>
      <td>0.2231</td>
    </tr>
  </tbody>
</table>

The models that utilize the spread as a feature are significantly better. Given the resources sportsooks have to calculate the spread, it makes sense that this would be one of the strongest predictors of who will win a given game. However, the SRS and regression based models also achieve a high accuracy, within a few percent of the sportsbooks. 
### This Season

Below are the results for this season after week 5.

<ul><li style="color: #C9B037"><span style="color: black">Best</span></li><li style="color: #D7D7D7"><span style="color: black">Second</span></li><li style="color: #AD8A56"><span style="color: black">Third</span></li></ul><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>accuracy</th>
      <th>brier score loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Spread + SRS</th>
      <td style="background-color:#C9B037">0.7440</td>
      <td style="background-color:#C9B037">0.1806</td>
    </tr>
    <tr>
      <th>Spread</th>
      <td style="background-color:#D7D7D7">0.7391</td>
      <td style="background-color:#D7D7D7">0.1826</td>
    </tr>
    <tr>
      <th>SRS + Decay</th>
      <td style="background-color:#AD8A56">0.7295</td>
      <td style="background-color:#AD8A56">0.2008</td>
    </tr>
    <tr>
      <th>Regression</th>
      <td>0.7053</td>
      <td>0.2035</td>
    </tr>
    <tr>
      <th>SRS</th>
      <td>0.7150</td>
      <td>0.2053</td>
    </tr>
  </tbody>
</table>

Both accuracy and brier score agree that the Spread + SRS model is better, delivering 0.5 percent higher accuracy. 
### Past 9 Super Bowls

The following table shows the probability given by each model that the winning team would win for each of the last 9 superbowls
<ul><li style="color: #C9B037"><span style="color: black">Best</span></li><li style="color: #D7D7D7"><span style="color: black">Second</span></li><li style="color: #AD8A56"><span style="color: black">Third</span></li></ul><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Winner</th>
      <th>Loser</th>
      <th>Spread</th>
      <th>Spread + SRS</th>
      <th>Regression</th>
      <th>SRS</th>
      <th>SRS + Decay</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015</th>
      <td>DEN</td>
      <td>CAR</td>
      <td>0.2895</td>
      <td>0.33</td>
      <td style="background-color:#AD8A56">0.4907</td>
      <td style="background-color:#D7D7D7">0.5017</td>
      <td style="background-color:#C9B037">0.5173</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>NE</td>
      <td>ATL</td>
      <td style="background-color:#C9B037">0.5769</td>
      <td style="background-color:#D7D7D7">0.556</td>
      <td style="background-color:#AD8A56">0.4679</td>
      <td>0.4635</td>
      <td>0.4286</td>
    </tr>
    <tr>
      <th>2017</th>
      <td>PHI</td>
      <td>NE</td>
      <td>0.3481</td>
      <td>0.3576</td>
      <td style="background-color:#D7D7D7">0.4653</td>
      <td style="background-color:#AD8A56">0.4635</td>
      <td style="background-color:#C9B037">0.4799</td>
    </tr>
    <tr>
      <th>2018</th>
      <td>NE</td>
      <td>LAR</td>
      <td style="background-color:#C9B037">0.5571</td>
      <td style="background-color:#D7D7D7">0.5525</td>
      <td>0.4007</td>
      <td>0.3962</td>
      <td style="background-color:#AD8A56">0.4268</td>
    </tr>
    <tr>
      <th>2019</th>
      <td>KC</td>
      <td>SF</td>
      <td>0.5262</td>
      <td style="background-color:#D7D7D7">0.5617</td>
      <td>0.5381</td>
      <td style="background-color:#AD8A56">0.542</td>
      <td style="background-color:#C9B037">0.5798</td>
    </tr>
    <tr>
      <th>2020</th>
      <td>TB</td>
      <td>KC</td>
      <td>0.4375</td>
      <td>0.4289</td>
      <td style="background-color:#D7D7D7">0.5953</td>
      <td style="background-color:#C9B037">0.6066</td>
      <td style="background-color:#AD8A56">0.5926</td>
    </tr>
    <tr>
      <th>2021</th>
      <td>LAR</td>
      <td>CIN</td>
      <td style="background-color:#C9B037">0.6984</td>
      <td style="background-color:#D7D7D7">0.6672</td>
      <td>0.4657</td>
      <td style="background-color:#AD8A56">0.5149</td>
      <td>0.5012</td>
    </tr>
    <tr>
      <th>2022</th>
      <td>KC</td>
      <td>PHI</td>
      <td style="background-color:#C9B037">0.4886</td>
      <td style="background-color:#D7D7D7">0.4537</td>
      <td style="background-color:#AD8A56">0.4033</td>
      <td>0.403</td>
      <td>0.4032</td>
    </tr>
    <tr>
      <th>2023</th>
      <td>KC</td>
      <td>SF</td>
      <td>0.4269</td>
      <td>0.4153</td>
      <td style="background-color:#AD8A56">0.4665</td>
      <td style="background-color:#D7D7D7">0.4814</td>
      <td style="background-color:#C9B037">0.5968</td>
    </tr>
  </tbody>
</table>

While the aggregate results above are clear that the spread based models are better, the trend is less clear for recent super bowls. The SRS + Decay model has been the best performing model 4 times, which is the same number as the spread model, and has predicted some unlikely outcomes. 
For instance, in 2015 the Panthers were heavy favorites (5.5 points) but ended up losing, which was predicted by the SRS + Decay model. The same is true for last years super bowl when the 49ers were favorites according to the sportsbooks but ended up losing as predicted by the SRS + Decay model. 

### Super Bowl Predictions

Given the analysis above about the strengths and weaknesses of each model, below are the probabilities of the Eagles winning this year's super bowl. 

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2024</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Spread</th>
      <td>0.446154</td>
    </tr>
    <tr>
      <th>Spread + SRS</th>
      <td>0.515683</td>
    </tr>
    <tr>
      <th>Regression</th>
      <td>0.720339</td>
    </tr>
    <tr>
      <th>SRS</th>
      <td>0.689655</td>
    </tr>
    <tr>
      <th>SRS + Decay</th>
      <td>0.67364</td>
    </tr>
  </tbody>
</table>

# Conclusion

Who will win this year's Super Bowl is a question of which model you want to trust. 
- If you trust the sportsbooks (which have provided the best predictions historically), the Chiefs have a 55 percent chance of winning
- If you trust statistical ratings models (like SRS) the Eagles have a 70 percent chance of winning
- If you trust the model combining these two sources (which has historically been about as good as the spread based model) the game is 50/50.

While the Eagles are a statistically better team with stronger wins, there are some factors (refs) that statistical models can't account for. Regardless of which model you choose to trust, this super bowl will be a tight game with lots of excitement for fans.  