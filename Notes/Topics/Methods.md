Sure, let's continue the conversation in English. Based on the sources, the following details the differences between [[DiCE]], [[GLANCE]], and other methods such as [[AReS]], *[[CET]] , and [[GLOBE-CE]].

### 1. DiCE (Diverse Counterfactual Explanations)

- **Scope:** **Local**. It is designed to explain a single prediction for one specific person.
- **Objective:** To generate a **diverse set** of counterfactual examples for an individual. This helps a user see multiple different paths to reach a favorable outcome (e.g., "you could get the loan by increasing your income **OR** by changing your education level").
- **Methodology:** It frames the search for explanations as an optimization problem that balances **proximity** (how close the change is to the original person) and **diversity** (how different the suggestions are from each other).

### 2. GLANCE (Global Actions in a Nutshell)

- **Scope:** **Global**. It focuses on the model's behavior across an entire population rather than one person.
- **Objective:** To find a **minimal set of actions** (typically around 3 or 4) that are effective for the largest possible number of people at the lowest cost.
- **Methodology:** It uses a unique **joint clustering** approach. It groups people not just by their characteristics (feature space) but also by the **similarity of the actions** they need to flip their outcome (action space). This allows it to find "compromise" actions that work for many different people.

### 3. Comparison with Other Global Methods

The researchers who developed GLANCE compared it against several previous global explainability frameworks:

|Method|Key Difference from GLANCE|Performance/Issues|
|:--|:--|:--|
|**GLOBE-CE**|Focuses on **directions** of change rather than fixed actions.|Can produce hundreds of "micro-actions," which makes it hard for humans to interpret. When restricted to a few actions, its effectiveness drops sharply.|
|**AReS / Fast AReS**|Uses a two-level **rule set** to summarize recourses.|Often fails to cover a large portion of the population; in many tests, it achieved less than 20% effectiveness.|
|**CET (CF Trees)**|Partitions the population using **decision trees** and assigns an action to each leaf.|Highly **computationally expensive**. It often times out or fails on large datasets (like "Adult Income") where GLANCE runs efficiently.|

### Summary of GLANCE’s Achievements

- **Pareto Dominance:** GLANCE achieved better results (higher effectiveness and lower cost) than its competitors in **57% of tested cases**.
- **Practicality:** Unlike methods like AReS, GLANCE consistently provided recourse to the majority of people, never falling below an 80% effectiveness threshold in experiments.
- **Human Preference:** A user study showed that people significantly prefer GLANCE because it provides **stable and robust** results with fewer, more understandable actions.