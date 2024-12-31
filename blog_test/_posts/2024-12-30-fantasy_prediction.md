---
layout: post2
title:  "Predicting Fantasy Football Scores"
date:   "2024-12-30"
---

How do you decide whether to start a player for your fantasy football team? You likely look at how many points that player has scored on average this season. You may pay special attention to the last few games to see if that player is on a hot streak. You also probably consider the strength of the opposing defense. From experience, you develop informal strategies for combining this information into predictions that allow you to draft players and to choose which players to start and which players to bench. Statistical modeling allows us to take these informal intuitions and fit them to data from past seasons to make predictions in a structured way.

In this article, I am going to share 3 different statistical models applied to 2 different fantasy football related problems. I'll begin by explaining the general process of "supervised learning," a statistical method for predicting an output variable based on input variables. I'll then apply this general process to the specific problem of predicting fantasy football scores. Next, I'll explain the different models and the problems we're using these models to solve. Finally, I'll evaluate each model's effectiveness.

## Supervised Learning 

Supervised learning is one type of machine learning. Machine learning fits statistical models to data. In supervised learning, the statistical models use input variables to predict an output variable. For example, let's say I'm trying to sell my house, and I want to know how much it might sell for. I have data about other houses in my neighborhood such as:
1. The size of the house in square feet
2. How many bedrooms the house has
3. How many bathrooms the house
If I know how much those houses sold for, I can fit a model to this data and predict how much my house might sell for based on this model. These input variables (size, bedrooms, and bathrooms) are called features.

To apply the general process of supervised learning, we need to choose a set of features and choose an outcome variable. We then split the data into a training set and a test set of data. We fit the model on the training set, and then evaluate our predictions on the test set. This process of splitting the data is analogous to how someone might prepare for a test. They have a set of practice problems (the training set) that they use to learn how to take the test. They then take the test (the test set) which is going to have problems they've never seen before. The problems on the test must differ from the practice problems if we want to evaluate whether the test taker understands the content or just memorized the practice problems. In supervised learning, "memorizing" is analogous to the concept of overfitting, which is when the model fits too close to the training data in a way that doesn't generalize to data that the model wasn't trained on. 

Once we have predicted outputs for our test set, we then compare these predictions to the actual values of outputs in the test set to determine how well the model fit. This comparison uses "metrics", which are mathematical ways of comparing a prediction and the actual outcome. To apply the general framework of supervised learning to the specific problem of predicting fantasy football scores, we need to define:
1. Features 
2. Target Variable
3. Training Set
4. Test Set
5. Models
6. Metrics

## Fantasy Football Score Prediction

The scores being predicted are Draftkings' fantasy scores. Draftkings' daily fantasy competitions assign a "salary" to each player allow competitors to draft teams of players with a total budget of $50k. The competitors then compete against each other to score the most fantasy points. Draftkings' fantasy scoring awards 1 points for receptions. We also limited our data to players that compete in Draftkings' main weekly contest (afternoon games on Sunday) and whose salary in that contest is above 4100 (to filter out players who likely won't score at all).
### Features

The features we will use to predict player performance are:
1. How many points has this player scored on average this season
2. How many points has this player scored in each of his last two games
3. How many points did their opponents allow, on average, to the player's position in the last two games
4. How many points, on average, has the player's current opponent given up
5. What is the player's fantasy salary

Below is an example of Patrick Mahomes' features in Week 9 of 2020 when he played the Panthers. The Panthers, on average, let the opponent's quarterback score 19.04 points. In the first nine weeks of the season, Mahomes had averaged 28 points per game. His most recent game (week 8) was against the Jets, who gave up 23.38 points on average to QBs, and Mahomes scored 39.64 points in this game. In week 7, Mahomes played the Broncos, who gave up 21.09 points to QBs on average and Mahomes scored 12 points against them. His fantasy salary for that week was 8100.

| Feature Name        | Feature Description                                                           | Feauture Value |
| :------------------ | :---------------------------------------------------------------------------- | -------------: |
| mean                | Average points the player has scored this season                              |        27.5125 |
| week -1 performance | How many points the player scored in their most recent game                   |          39.64 |
| week -2 performance | How many points the player scored 2 games ago                                 |           12.0 |
| opponent average -1 | Average points given up to the players position by their most recent oppenent |          23.38 |
| opponent average -2 | Average points given up by the opponent 2 games ago                           |          21.09 |
| opponent average    | Average points given up by the opponent they are about to face                |          19.04 |
| salary              | Draftkings salary divided by 100                                              |           8100 |
| is Quarterbacks     | True if the player is a quarterback                                           |              1 |
| is Running Backs    | True if the player is a running back                                          |              0 |
| is Tight Ends       | True if the player is a tight end                                             |              0 |
| is Wide Receivers   | True if the player is a wide receiver                                         |              0 |

### Output Variable

There are two classes of supervised learning problems whose output variables differ. Regression problems aim to predict a continuous value (such as the price of a house). Classification problems aim to classify a data point into 1 of multiple classes (such as predicting whether a house will fall into foreclosure), giving probabilities that the data point will be in each class. We will look at problems of both types in this article.

The regression problem predicts how many fantasy points a player will score. The classification problem predicts the probability that a player will score more than a certain number of points. 

Both problems provide unique insights into potential performance. The regression problem tells us on average what we can expect a particular player to score. The classification problem gives us additional insight into what the range of potential outcomes a player might have. For instance, let's say we have two players who both have a predicted fantasy score of 12 points. However, player 1 always scores between 10 and 14 points while player 2 could score anywhere from 0 to 24 points. If each player's score is uniform over his range, each player has an expected score of 12 points but wildly different potential outcomes. The classification problem output would reflect this difference. Player 1 would have a 0 percent probability of scoring over 20 points, while player 2 would have a much higher probability of scoring over 20 points. Thus, the output of the classification problem allows us to determine whether a player is a consistent and reliable fantasy player or a potential breakout player who could also score very few points (boom or bust).

In summary, the problems being solved are:
1. Regression (what will a player score)
2. Classification (what is the probability that a player will score more than a certain number of points) for point thresholds of 10, 15 and 20 points.


### Training and Test Set

The training set will be the 2017-2019 seasons after week 5. The test set will be the 2020 season after week 5. We don't use data before week 5 so players have multiple weeks of performance data and so averages have stabilized and aren't dominated by outliers. This split in the data serves as a simulation of what predictions would've been in 2020 if, at the beginning of the 2020 season, we had access to the 2017 to 2019 results. The training set has about 6000 data points and the test set has about 2000.
### Models

I will give a brief description of each model. This will be a high-level overview, but the links provided will go into greater detail.

The models of interest are:
1. Linear Model
2. Gradient Boosting Model
3. Nearest Neighbors Model

**Linear Model**

The linear model is one of the simplest statistical models. It makes a prediction by multiplying each of the features by a coefficient and summing these values together. These coefficients represent the "value" each feature contributes towards the output. Returning to the example of predicting house prices, a linear model makes intuitive sense because each feature adds some amount of value to the house (there is some dollar value added by each bedroom and each bathroom). An example linear model might predict a house's price as 10 times the square footage ($10 per square foot), plus 30k times the number of bedrooms (30k per bedroom), plus 10k times the number of bathrooms (10k per bathroom). This would mean a 1500 sq ft 3 bed 2 bath would have an estimated price of 10 times 1500 plus 30k time 3 plus 10k times 2, which is 125k. Using supervised learning, we don't have to guess at what the value (coefficient) of each feature is. The linear model learns these coefficients from the training data by minimizing the squared difference between the models predictions and the actual values. We can use this model to predict probabilities (values between 0 and 1) by applying a [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) function to the output of the linear model. 

**Gradient Boosting**

[Gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting) is a method that combines the results of many simple models to get a final prediction. This is the most complex model used in this article, but also one of the most powerful (and as you'll see later, the best model for our particular problem). Conceptually, gradient boosting iteratively improves its predictions by fitting multiple models, with each model focusing on the data points with the largest errors in the previous round of model fitting.  

**Nearest Neighbors**

Nearest neighbors is another simple statistical model. Based on the graph below, where values between x=4.5 and x=5.5 have been removed, If I asked you to predict the value when x=5, how might you approach this?. 

![Image](/images/fig1.png)

One reasonable way might be to look at the values between x=4 and x=6 and take the average of those points' y values. This is the same logic used in the nearest neighbors model. Nearest neighbors finds the data points in the training set whose features are "closest" to a given input's features and uses those data points to make predictions. To find the "closest" point, we need to define a distance metric. For the problems in this article, we learn the distance metrics based on the training data. I won't go into the distance metrics in greater detail, but the approaches used here are [Neighborhood Component Analysis](https://scikit-learn.org/stable/modules/neighbors.html#nca)  for classification and [MLKR](https://contrib.scikit-learn.org/metric-learn/supervised.html#mlkr) for regression.

### Metrics

**Regression Metrics**

The two regression metrics used are mean squared error and mean absolute error. Squared error is the square of difference between the actual value and our model's prediction. Absolute error is the absolute value of the difference between the actual value and the model's prediction. For instance, if I predict Patrick Mahomes will score 10 points and he scores 12, then the squared error of my prediction is 2 squared or 4 and the absolute error is 2. The mean squared and mean absolute errors simply take the errors for all players in the test set and averages across these.

**Classification Metrics**

For classification results, we use one metric based on the predicted probabilities and several binary metrics based on the actual class being predicted. The probability based metric is the [brier score](https://scikit-learn.org/stable/modules/model_evaluation.html#brier-score-loss). This is a score between 0 (best) and 1 (worst). This score is the square of the (1-predicted probability) of an event happening. So if a model says Patrick Mahomes has a 70 percent chance of scoring over 20 points and he does score over 20 points, then the brier score for this prediction is (1-0.7) squared, which is 0.09. This score rewards both predicting the correct label and having a high confidence in your predictions (the brier score would be lower if the model said Mahomes had a 90 percent chance instead of a 70 percent chance of scoring over 20). The brier score metric below averages these values for all predictions in the test set.

The binary metrics take the probabilities and convert these to binary predictions. Instead of a probability that Patrick Mahomes will score over 20 points, the output is just a prediction of will he or won't he. We can convert probabilities to predictions by predicting a player will score more than the threshold if the probability of them doing so is greater than 50 percent. We then compare these binary predictions to what actually happened using the following metrics (in these metrics, a "true positive" is an instance where the model predicts a player will score above a certain number of points and the player actually does score above that amount)
- Accuracy: What percent of predictions were correct 
- Recall: True positives divided by total positives
	- For the problem of predicting how many players will score over 20 points, this metric is the number of players correctly predicted to score over 20 divided by the number who actually did score over 20. If there were 100 players who scored over 20 points, and our model predicted 30 of those players would score over 20, the recall would be 30 percent.
- Precision: True positives divided by total predicted positives
	- In the case above, where the model has 30 true positives, if the model predicted 50 players would score above 20, then the precision is 30/50=60 percent. A high-precision model doesn't mistakenly say a player will score over 20 (so if the model predicts a player will score over 20, he most likely will)
- Balanced Accuracy: Variant of accuracy that weights data points so that classes are equally represented.
	- This gives a clearer view of accuracy when the sample is unbalanced. For instance, there are a lot more players who score less than 20 points than players who score over 20 points, so the balanced accuracy accounts for this by giving more weight to samples where the player scores over 20 points.

## Results

Below are tables showing average results for each problem and model.

**Regression**

|                     |  Linear | Boosting | Neighbors |
| :------------------ | ------: | -------: | --------: |
| Mean Squared Error  |  69.124 |   68.766 |   70.9935 |
| Mean Absolute Error | 6.60797 |   6.5496 |   6.67841 |

The best regression model is boosting, followed closely by the linear model and, more distantly, by the nearest neighbors model.

**Greater than 10 points**

|                   | Linear | Boosting | Neighbors |
| :---------------- | -----: | -------: | --------: |
| Brier Score       | 0.2049 |   0.2036 |    0.2082 |
| Balanced Accuracy | 0.6634 |   0.6541 |    0.6347 |
| Accuracy          | 0.6832 |   0.6753 |    0.6642 |
| Recall            | 0.7675 |   0.7657 |    0.7897 |
| Precision         |  0.719 |   0.7109 |    0.6905 |

The best model for predicting whether a player will score over 10 points is the gradient boosting model because it has the lowest brier score. However, the linear model achieves a higher accuracy with both a higher recall and higher precision, suggesting it provides better labels than the boosting model.

**Greater than 15 Points**

|                   | Linear | Boosting | Neighbors |
| :---------------- | -----: | -------: | --------: |
| Brier Score       | 0.1946 |   0.1938 |    0.2024 |
| Balanced Accuracy | 0.6742 |   0.6639 |     0.649 |
| Accuracy          | 0.7101 |    0.699 |    0.6885 |
| Recall            | 0.5074 |   0.5007 |    0.4657 |
| Precision         | 0.6732 |   0.6515 |    0.6419 |

The best model for predicting whether a player will score over 15 points is the gradient boosting model because it has the lowest brier score. However, the linear model achieves a higher accuracy with both a higher recall and higher precision, suggesting it provides better labels than the boosting model.

**Greater than 20 Points**

|                   | Linear | Boosting | Neighbors |
| :---------------- | -----: | -------: | --------: |
| Brier Score       | 0.1526 |   0.1535 |     0.159 |
| Balanced Accuracy | 0.6087 |   0.6033 |    0.5501 |
| Accuracy          | 0.7835 |   0.7835 |    0.7703 |
| Recall            | 0.2762 |   0.2606 |    0.1314 |
| Precision         | 0.5933 |      0.6 |    0.5673 |

The best model for predicting whether a player will score over 20 points is the linear model because it has the lowest brier score. The linear model achieves a higher accuracy but has worse precision than the boosting model.

The three models are similar in performance with the gradient boosting model performing the best of the three in most cases. While the gradient boosting provides better regression estimates and better brier scores by providing more confident probabilities, the linear model can more accurately label data points. You can view the predictions made by these models both in 2020 and for week 17 of the 2024 season in the dashboard below. The predictions don't take into account injuries or inactive players, so it will make predictions as if the player wasn't injured. You can also dig more deeply using the complete dashboard linked here.

<iframe src="https://fantasyfootballprediction-bed7fctpmnoa7zdtzsrfxk.streamlit.app//?embed=true" height="720px" width="100%" style="border:none;"></iframe>

