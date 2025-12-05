# Assignment 6 Part 2 - Writeup
Patrick Nyman
---

## Question 1: Feature Importance

Based on your house price model, rank the four features from most important to least important. Explain how you determined this ranking.

**YOUR ANSWER:**
1. Most Important: Bedrooms
2. Bathrooms
3. Age
4. Least Important: SquareFeet

**Explanation:**
This was based on the the ranking of importance which the program gave. Bedrooms had a value of 6648.97, bathrooms had a value of 3858.90, age had a value of 950.35, and square feet had the lowest value, at 121.11. The value of these numbers described the amount of importance the variable had on the graph, determined by taking the absolute value of their coefficients.



---

## Question 2: Interpreting Coefficients

Choose TWO features from your model and explain what their coefficients mean in plain English. For example: "Each additional bedroom increases the price by $___"

**Feature 1:**
Each additional square foot increases the price of the house by 121.11 dollars.

**Feature 2:**
Each additional year of age decreases the price of the house by 950.35 dollars.

---

## Question 3: Model Performance

What was your model's R² score? What does this tell you about how well your model predicts house prices? Is there room for improvement?

**YOUR ANSWER:**
The R2 score of the model was 0.9936. This means that the model does a really great job at predicting the prices. There is always room for improvement; it can always imrprove by training on more data.



---

## Question 4: Adding Features

If you could add TWO more features to improve your house price predictions, what would they be and why?

**Feature 1:**
Floor levels

**Why it would help:**
A house with more floors will cost more, because people are willing to pay more for increased space. This would cause a linear relationship between floors and price.

**Feature 2:**
District

**Why it would help:**
A house in a more wealthy district will cost more than a house in a poorer area. Districts can be rated on a scale, allowing the correlation to be graphed.

---

## Question 5: Model Trust

Would you trust this model to predict the price of a house with 6 bedrooms, 4 bathrooms, 3000 sq ft, and 5 years old? Why or why not? (Hint: Think about the range of your training data)

**YOUR ANSWER:**
I would trust the model to make this prediction because the values are very close in value to the data that the model was trained on. The variable for age is already included in the training data. The rest of the variables are within the range of the training data where it would not influence the line of best fit greatly.

