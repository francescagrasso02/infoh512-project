# DiCE Methodology
> Reference document for the project

## 1. What is DiCE?
Diverse Counterfactual Explainations is a Python library developed by Mothilal, Sharma & Tan in 2020 that generates *multiple*, *diverse* counterfactual
explanations for any machine learning model.
Model prediction with counterfactual explanations provides like *"what if?"* hypothetical examples for model output and can be a useful complement to other explanation methods.

**Crucial elements of DiCE**
- *Counterfactual explainations*: in order to gain understanding why the model arrived to a certain prediction, DiCE generates similar case to the one in instance but with a different conclusion.
- *Diversity*: counterfactual explanations may reveal presence of biases within the model
- *Model agnostic*: not model specific, therefore it can generalize to any kind of ML algos

## 2. Core idea related to our project
Given an employee predicted to leave with `Attrition=1`, the DiCE reply would be:
> *"What is the minimum set of changes the employee could make so that the model predicts they will stay?"*

For example (https://medium.com/@susmar1304/decoding-machine-learning-decisions-the-power-of-dice-ml-counterfactuals-49b699712972)
Using the same approach as for the famous Iris dataset but applying it to our:

| Feature | Original | Counterfactual 1 | Counterfactual 2 |
|---|---|---|---|
| OverTime | Yes | No | No |
| Department | Sales | Sales | R&D |
| WorkLifeBalance | 1 | 3 | 1 |
| Prediction | Leaves | Stays | Leaves |

What's new is that this algo is generating *diverse* counterfactuals so employees have multiple actionable paths.

## 3. Implementation in our pipeline

So we have the IBM HR Dataset (1,470 employees)
--> Preprocessing: encode categorical, scale continuous, check eventual class imnbalance and define a way to handle it
--> Train ML Classifier (as Matteo said, no need to find a perfect method, just one that works)
--> Attrition prediction (0/1)
--> Subclass: `Attrition=1` as we are interested in finding alternative scenarios for this case
--> Define features list (there are for sure some features we cannot use as parameters that may change for instance the gender and the age)
--> Generate counterfactuals per employee
--> Aggregate counterfactuals by demographic group
--> Apply FACTS metrics

## 4. Important things we need to define
1. Which features can be changed? Age and Gender are immutable for instance also because they are going to be use as group references for the second part
2. How many counterfactuals per employee? 3-4 might be an accessible number
3. Which DICE method? [need to check the documentation]
4. Inside the DiCE methodology: proximity vs diversity weight
5. 


### 4a. Specificities regarding ML classifier
Boosting and random forest are for definition models that are not so transparent for humans as it is not easy to understand why one result is obtained from given model in an interpretable way (XAI from DataCamp).

### 4b. Specificities regarding DiCE model 
    > https://medium.com/analytics-vidhya/dice-ml-models-with-counterfactual-explanations-for-the-sunk-titanic-30aa035056e0
- a. **Proximity (default weight: 0.5) vs diversity (default 1.0)** are changeable values when we generate counterfactual examples. 
  To see how these weights generate different sets of various counterfactual explanations one idea would be to put a iteration variable (for loop with different parameters)
- b. **Feature weight**: dictionary argument we can give for each umerical features to configure its difficulty to change the feature value
- c. **features_to_vary** list: some features cannot be varied
