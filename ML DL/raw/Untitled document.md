   
   
   
   
Empirical Risk Minimization (ERM) is the foundational principle in machine learning that guides a model to learn by finding the parameters that minimize the average error (loss) on a training dataset. It serves as a practical, calculable substitute for finding the true, unattainable minimum error across all possible real-world data.   
   
![1\. Loss Function L: Quantifies the error between the true output y; and the predicted output h(x¡). 2. True Risk R(h): R(h) = E(x,y)\~p\[{(y,h(x))\] (where P is the true, unknown data distribution) 3. Empirical Risk R(h): Ñ(h)= - 21 \_\_ {(yi, h(x;)) 4. ERM Objective: h\* = argminhE\*R(h) (We seek the model h that yields the absolute lowest average training error) GeeksforGeeks +1][image1]  
   
   
   
   
   
   
   
   
The chi-square test and t-test are both statistical tests used in hypothesis testing, but they are used in different situations and for different types of data.   
**Chi-Square Test:** 

1. **Purpose**: The chi-square test is used to determine whether there is a significant association between categorical variables.   
2. **Types**:   
   * **Chi-square test of independence**: Tests whether there is a significant association between two categorical variables.   
   * **Chi-square test of goodness of fit**: Tests whether observed categorical data fit a theoretical distribution.   
3. **Data Type**: It is used with categorical data, where data points fall into specific categories or groups.   
4. **Assumptions**:   
   * The variables are categorical.   
   * The observations are independent.   
   * The expected frequency count in each cell of the contingency table is at least 5\.   
5. **Output**: The chi-square test produces a test statistic and a p-value that indicates whether the observed association between variables is statistically significant. 

**T-Test:** 

1. **Purpose**: The t-test is used to determine whether there is a significant difference between the means of two groups.   
2. **Types**:   
   * **Independent t-test**: Compares the means of two independent groups.   
   * **Paired t-test**: Compares the means of two related groups (e.g., before and after measurements).   
3. **Data Type**: It is used with interval or ratio data, where the data points are numerical and continuous.   
4. **Assumptions** (for independent t-test):   
   * The data are approximately normally distributed within each group.   
   * The variances of the populations are approximately equal.   
   * The observations are independent.   
5. **Output**: The t-test produces a test statistic (t-value) and a p-value that indicates whether the difference in means between groups is statistically significant. 

**Key Differences:** 

* **Data Type**: Chi-square tests categorical data, while t-tests are used with numerical (continuous) data.   
* **Purpose**: Chi-square tests for association between categorical variables, while t-tests compare means of numerical variables.   
* **Assumptions**: Chi-square test assumes categorical data and independence of observations; t-test assumes normally distributed data and independence of observations (for independent t-test). 

In summary, choose between a chi-square test and a t-test based on the type of data you have (categorical vs. numerical) and the specific hypothesis you want to test (association between variables vs. difference in means). Each test has its own assumptions and applications, ensuring you choose the most appropriate one for your analysis.   
   
   
   
   
   
   
   
Hypothesis testing is a fundamental statistical method used to make inferences about a population based on sample data. It involves evaluating two competing hypotheses about the population parameter of interest: the null hypothesis (H₀) and the alternative hypothesis (H₁ or Hₐ).   
**Steps in Hypothesis Testing:** 

1. **Formulate Hypotheses**:   
   * **Null Hypothesis (H₀)**: This is the default assumption or the status quo. It represents no effect, no difference, or no relationship in the population. It is what you aim to test against.   
   * **Alternative Hypothesis (H₁ or Hₐ)**: This is the opposite of the null hypothesis. It represents what you suspect to be true or hope to demonstrate. It suggests an effect, difference, or relationship exists in the population.   
     Example:   
   * H₀: There is no difference in mean scores between Group A and Group B.   
   * H₁: There is a difference in mean scores between Group A and Group B.   
2. **Choose a Significance Level (α)**:   
   * The significance level (α) is the probability threshold below which you reject the null hypothesis. Commonly used values are 0.05 (5%) and 0.01 (1%).   
   * It represents the probability of rejecting the null hypothesis when it is actually true (Type I error).   
3. **Select a Statistical Test**:   
   * The choice of statistical test depends on the nature of your data (categorical or numerical) and the specific hypotheses being tested.   
   * Examples include t-tests, chi-square tests, ANOVA, correlation tests, etc.   
4. **Collect and Analyze Data**:   
   * Collect a sample from the population of interest.   
   * Calculate the appropriate test statistic (e.g., t-value, chi-square statistic) using sample data.   
5. **Compare Results to the Null Hypothesis**:   
   * Determine the p-value: The p-value is the probability of obtaining results as extreme as the observed data, assuming the null hypothesis is true.   
   * If the p-value is less than or equal to α (typically 0.05), reject the null hypothesis in favor of the alternative hypothesis.   
   * If the p-value is greater than α, fail to reject the null hypothesis (note: this does not mean "accept" the null hypothesis).   
6. **Draw Conclusions**:   
   * Based on the results, draw conclusions about the population parameter.   
   * If you reject the null hypothesis, you may conclude that there is sufficient evidence to support the alternative hypothesis.   
   * If you fail to reject the null hypothesis, you may conclude that there is not enough evidence to support the alternative hypothesis. 

**Key Concepts in Hypothesis Testing:** 

* **Type I Error**: Rejecting the null hypothesis when it is actually true (false positive).   
* **Type II Error**: Failing to reject the null hypothesis when it is actually false (false negative).   
* **Power**: Probability of correctly rejecting the null hypothesis when it is false. 

**Example:**   
Suppose you want to test if a new drug treatment is more effective than a placebo. Your hypotheses would be: 

* H₀: The mean effectiveness score for the drug treatment is the same as for the placebo.   
* H₁: The mean effectiveness score for the drug treatment is higher than for the placebo. 

You would collect data, perform a statistical test (e.g., independent t-test), obtain a p-value, compare it to your chosen significance level (e.g., α \= 0.05), and then draw conclusions based on whether you reject or fail to reject the null hypothesis.   
Hypothesis testing is crucial in scientific research, allowing researchers to make informed decisions and draw reliable conclusions based on data analysis.   
   
   
   
   
   
   
   
Calculating the p-value is a critical step in hypothesis testing, as it helps determine the strength of evidence against the null hypothesis. The exact method to compute the p-value depends on the type of statistical test you are conducting. Here’s a general outline of how p-values are computed for common statistical tests:   
**1\. Z-Test (for a single mean):** 

* **Formula**:   
  Z=xˉ−μσn\\text{Z} \= \\frac{\\bar{x} \- \\mu}{\\frac{\\sigma}{\\sqrt{n}}}Z=nσxˉ−μ   
  where:   
  * xˉ\\bar{x}xˉ is the sample mean,   
  * μ\\muμ is the population mean (under the null hypothesis),   
  * σ\\sigmaσ is the population standard deviation (known),   
  * nnn is the sample size.   
* **P-Value Calculation**:   
  * For a two-tailed test: p-value=2×P(Z\>∣Z∣)\\text{p-value} \= 2 \\times P(Z \> |\\text{Z}|)p-value=2×P(Z\>∣Z∣)   
  * For a one-tailed test (directional hypothesis): p-value=P(Z\>Z)\\text{p-value} \= P(Z \> \\text{Z})p-value=P(Z\>Z) or p-value=P(Z\<Z)\\text{p-value} \= P(Z \< \\text{Z})p-value=P(Z\<Z), depending on the direction of the alternative hypothesis. 

**2\. T-Test:** 

* **Independent T-Test**:   
  * Formula for t-statistic: t=xˉ1−xˉ2s12n1+s22n2\\text{t} \= \\frac{\\bar{x}\_1 \- \\bar{x}\_2}{\\sqrt{\\frac{s\_1^2}{n\_1} \+ \\frac{s\_2^2}{n\_2}}}t=n1s12+n2s22xˉ1−xˉ2 where:   
    * xˉ1,xˉ2\\bar{x}\_1, \\bar{x}\_2xˉ1,xˉ2 are the sample means,   
    * s1,s2s\_1, s\_2s1,s2 are the sample standard deviations,   
    * n1,n2n\_1, n\_2n1,n2 are the sample sizes.   
* **P-Value Calculation**:   
  * Degrees of freedom (dfdfdf) is calculated based on the sample sizes n1n\_1n1 and n2n\_2n2.   
  * Use statistical tables, software, or online calculators to find the p-value associated with the calculated t-statistic and degrees of freedom. 

**3\. Chi-Square Test:** 

* **Chi-Square Statistic**:   
  χ2=∑(O−E)2E\\chi^2 \= \\sum \\frac{(O \- E)^2}{E}χ2=∑E(O−E)2   
  where:   
  * OOO is the observed frequency,   
  * EEE is the expected frequency under the null hypothesis.   
* **P-Value Calculation**:   
  * Degrees of freedom (dfdfdf) is determined by the number of categories minus 1\.   
  * Use statistical tables, software, or online calculators to find the p-value associated with the calculated χ2\\chi^2χ2 statistic and degrees of freedom. 

**4\. ANOVA:** 

* **F-Statistic**:   
  F=Between-group variabilityWithin-group variabilityF \= \\frac{\\text{Between-group variability}}{\\text{Within-group variability}}F=Within-group variabilityBetween-group variability   
* **P-Value Calculation**:   
  * Degrees of freedom for between-group and within-group variability are calculated based on the number of groups and the total sample size.   
  * Use statistical tables, software, or online calculators to find the p-value associated with the calculated F-statistic and degrees of freedom. 

**General Steps for Finding P-Value:** 

1. **Calculate the Test Statistic**: Compute the appropriate test statistic based on your data and the hypothesis being tested.   
2. **Determine the Distribution**: Identify the distribution (Z, t, χ2\\chi^2χ2, F, etc.) that corresponds to your test statistic.   
3. **Find the P-Value**: Use statistical tables, software, or online calculators to find the p-value associated with your calculated test statistic and degrees of freedom (if applicable).   
4. **Interpret the P-Value**: Compare the obtained p-value with the significance level (α\\alphaα) chosen for the test. If the p-value is less than or equal to α\\alphaα, reject the null hypothesis. If the p-value is greater than α\\alphaα, fail to reject the null hypothesis.   
5. **Report**: State your conclusion based on the p-value and interpret the findings in the context of your study. 

When performing hypothesis testing, accurate computation and interpretation of the p-value are crucial for making informed decisions and drawing valid conclusions from your data.   
   
   
   
   
   
   
   
   
   
**R-squared (R²)** and **adjusted R-squared (R²\_adjusted)** are both metrics used to evaluate the goodness of fit of a regression model, but they serve different purposes and account for different aspects of model performance.   
**R-squared (R²):** 

* **Definition**: R-squared is a statistical measure that represents the proportion of the variance in the dependent variable that is explained by the independent variables in the model.   
* **Range**: R-squared values range from 0 to 1\.   
* **Interpretation**:   
  * A higher R-squared value indicates that a larger proportion of the variance in the dependent variable is explained by the independent variables.   
  * R-squared does not indicate whether the model itself is adequate; it only shows how well the model fits the data.   
* **Formula**:   
  R2=1−SSresidualSStotalR^2 \= 1 \- \\frac{SS\_{\\text{residual}}}{SS\_{\\text{total}}}R2=1−SStotalSSresidual   
  * SSresidualSS\_{\\text{residual}}SSresidual is the sum of squared residuals (errors) of the regression model.   
  * SStotalSS\_{\\text{total}}SStotal is the total sum of squares of the dependent variable.   
* **Limitations**:   
  * R-squared increases with the addition of more independent variables, even if those variables are not significant. Therefore, it does not penalize for overfitting. 

**Adjusted R-squared (R²\_adjusted):** 

* **Definition**: Adjusted R-squared is a modified version of R-squared that adjusts for the number of predictors in the model.   
* **Purpose**: It penalizes for adding independent variables that do not improve the model significantly.   
* **Range**: Adjusted R-squared values can be negative and typically range from −∞-\\infty−∞ to 1\.   
* **Formula**:   
  Radjusted2=1−(1−R2)⋅(n−1)n−k−1R^2\_{\\text{adjusted}} \= 1 \- \\frac{(1 \- R^2) \\cdot (n \- 1)}{n \- k \- 1}Radjusted2=1−n−k−1(1−R2)⋅(n−1)   
  * nnn is the number of observations.   
  * kkk is the number of independent variables (predictors) in the model.   
* **Interpretation**:   
  * Adjusted R-squared increases only if the new term improves the model more than would be expected by chance.   
  * It generally provides a more realistic assessment of the model’s fit, especially when dealing with multiple predictors.   
* **Use**: Adjusted R-squared is particularly useful when comparing regression models with different numbers of predictors or when trying to determine whether adding more variables to the model improves its explanatory power significantly. 

**Key Differences:** 

* **Penalization**: R-squared does not penalize for adding unnecessary variables, whereas adjusted R-squared penalizes for adding variables that do not improve the model.   
* **Interpretation**: Adjusted R-squared tends to be lower than R-squared when predictors are added to the model that do not improve the fit.   
* **Purpose**: R-squared is primarily used to assess how well the model fits the data, whereas adjusted R-squared is used to assess how well the model fits the data while considering the number of predictors. 

In summary, while R-squared provides a measure of how well the regression model explains the variation in the dependent variable, adjusted R-squared offers a more conservative estimate by considering the model’s complexity due to the number of predictors. Adjusted R-squared is often preferred for evaluating the quality and relevance of regression models, especially when comparing models with different numbers of predictors.   
   
   
   
   
   
   
   
   
   
   
Positional embeddings 

* Transformers do not know the order of words   
* word embedding is added with positional embedding to add information of position   
* Position embeddings remain same irrespective of sequence length and input   
* The values should not be very large otherwise they shift the embeddings too far from their space.   
  * alternated by sine and cosine in increasing frequencies for each value in position vector 

   
   
   
   
   
   
   
   
   
   
**Ensemble learning techniques** \- combines multiple weak learners.   
   
   
   
**Bagging** 

1. **Method**:   
   * Each model in the ensemble is trained independently using a **random subset** of the data with replacement (bootstrapping).   
   * The final prediction is typically made by averaging the predictions (for regression) or majority voting (for classification).   
2. **Objective**:   
   * Reduce variance and prevent overfitting by averaging multiple models.   
3. **Examples**:   
   * Random Forest: An ensemble of decision trees where each tree is trained on a different bootstrap sample of the data.   
4. **Training Process**:   
   * Generate multiple bootstrap samples from the training dataset.   
   * Train a separate model on each bootstrap sample.   
   * Combine the predictions from all models.   
5. **Strengths**:   
   * Works well with high variance models like decision trees.   
   * Simple and parallelizable since each model is trained independently.   
6. **Weaknesses**:   
   * Does not focus on difficult-to-predict instances.   
   * Might not significantly improve performance if base learners are strong and not very sensitive to data variations. 

**Boosting** 

1. **Method**:   
   * Models are trained sequentially, each new model focusing on correcting the errors made by the previous models.   
   * Weights are assigned to each instance, with more weight given to misclassified instances.   
2. **Objective**:   
   * Reduce bias and variance by focusing on difficult instances and combining weak learners into a strong learner.   
3. **Examples**:   
   * AdaBoost: Adjusts the weights of incorrectly classified instances and combines the weak learners' predictions through a weighted majority vote.   
   * Gradient Boosting: Sequentially adds models that minimize a loss function using gradient descent, often using decision trees as base learners.   
4. **Training Process**:   
   * Initialize model weights and train the first model.   
   * Adjust weights based on the performance of the previous model.   
   * Train the next model on the weighted data.   
   * Repeat until the desired number of models is trained.   
   * Combine the predictions of all models, often using weighted voting.   
5. **Strengths**:   
   * Effective at reducing both bias and variance.   
   * Can achieve high accuracy by focusing on difficult instances.   
6. **Weaknesses**:   
   * More prone to overfitting, especially with noisy data.   
   * Training is sequential and less parallelizable compared to bagging.   
   * More computationally intensive due to the iterative training process. 

| Feature  | Bagging  | Boosting  |
| :---- | :---- | :---- |
| Method  | Bootstrap sampling, independent model training  | Sequential model training, weighted adjustments  |
| Objective  | Reduce variance, prevent overfitting  | Reduce bias and variance, focus on hard cases  |
| Training  | Parallelizable, multiple bootstrap samples  | Sequential, models focus on previous errors  |
| Combination  | Averaging (regression) or voting (classification)  | Weighted voting or summation  |
| Examples  | Random Forest  | AdaBoost, Gradient Boosting  |
| Strengths  | Simple, reduces overfitting, works well with high variance models  | High accuracy, reduces bias and variance  |
| Weaknesses  | Might not improve strong learners significantly  | Prone to overfitting, computationally intensive  |

   
   
   
   
   
   
   
   
XGBoost (Extreme Gradient Boosting) and Random Forest are both popular ensemble learning algorithms, but they have distinct characteristics and are suited to different types of problems. Here's a comparison:   
**Random Forest:** 

1. **Algorithm**:   
   1. Consists of multiple decision trees, typically trained with the bagging method.   
   2. Each tree is built independently, and the final prediction is made by averaging (regression) or voting (classification).   
   3. **Strengths**:   
      1. **Robust to Overfitting**: The averaging of multiple trees helps to reduce overfitting.   
      2. **Handles Missing Values**: Can handle missing data well through surrogate splits.   
      3. **Easy to Use**: Requires less parameter tuning and is generally easier to implement.   
      4. **Weaknesses**:   
         1. **Slower Predictions**: With many trees, the prediction time can be slower compared to a single model.   
         2. **Less Interpretability**: Although it can provide feature importance, the overall model is less interpretable.   
         3. **Use Cases**:   
            1. Works well for a variety of problems, especially when you need a quick and robust model without extensive parameter tuning.   
            2. Effective for both classification and regression tasks.   
         4. **XGBoost:**   
            1. **Algorithm**:   
               1. An implementation of gradient boosting that builds trees sequentially.   
               2. Each new tree corrects errors made by the previous trees.   
               3. **Strengths**:   
                  1. **Performance**: Often achieves higher accuracy compared to Random Forest, especially on structured/tabular data.   
                  2. **Speed and Efficiency**: Optimized for speed and performance with parallelization and hardware optimization.   
                  3. **Feature Importance**: Provides detailed insights into feature importance.   
                  4. **Regularization**: Includes L1 (Lasso) and L2 (Ridge) regularization to prevent overfitting.   
                  5. **Weaknesses**:   
                     1. **Complexity**: Requires more careful parameter tuning to achieve optimal performance.   
                     2. **Overfitting**: Can overfit if not properly regularized, especially on small datasets.   
                     3. **Use Cases**:   
                        * Particularly effective for structured data where high performance is needed.   
                        * Preferred in many competitive machine learning challenges due to its accuracy and efficiency.   
                     4. **Summary:**   
                        * **Random Forest** is a solid choice for quickly building a reliable model with minimal parameter tuning. It is robust and works well out-of-the-box.   
                        * **XGBoost** requires more effort in tuning but can achieve superior performance, particularly on large and complex datasets.   
                     5. Choosing between them often depends on the specific requirements of your project, such as the need for speed, accuracy, interpretability, and ease of use. 

   
   
   
   
   
   
   
   
   
   
Boosting types 

* Gradient boosting   
* AdaBoost   
* XGBoost 

   
   
In XGBoost, the "X" stands for "eXtreme." The name reflects the algorithm's focus on providing extreme (high) performance and efficiency in machine learning tasks, particularly through its optimized implementation of gradient boosting.   
   
 

| [AdaBoost](https://www.geeksforgeeks.org/implementing-the-adaboost-algorithm-from-scratch/)  | Gradient Boosting  |
| :---: | :---: |
| During each iteration in AdaBoost, the weights of incorrectly classified samples are increased, so that the next weak learner focuses more on these samples.  | Gradient Boosting updates the weights by computing the negative gradient of the loss function with respect to the predicted output.  |
| AdaBoost uses simple decision trees with one split known as the decision stumps of weak learners.  | Gradient Boosting can use a wide range of base learners, such as decision trees, and linear models.  |
| AdaBoost is more susceptible to noise and outliers in the data, as it assigns high weights to misclassified samples  | Gradient Boosting is generally more robust, as it updates the weights based on the gradients, which are less sensitive to outliers.  |

   
   
   
   
   
   
Kullback-Leibler (KL) divergence 

* Measure of the difference between two probability distributions, can be used to overlap the two distributions (initial LM output vs. tuned LM output).   
* KL divergence is a measure of how one probability distribution diverges from a second, expected probability distribution. It quantifies the difference between two probability distributions. 

   
   
   
   
   
   
   
   
   
   
LLAMA Chatbot   
   
   
   
   
   
   
   
During the pre-training of the LLaMA (Large Language Model Meta AI) models, the loss function used is typically the **cross-entropy loss**. This loss function is standard for training language models and measures the difference between the predicted probability distribution over the vocabulary and the actual distribution, which is a one-hot encoded vector representing the correct next token.   
Here's a bit more detail on how it works: 

1. **Cross-Entropy Loss**:   
   1. **Objective**: The goal is to minimize the difference between the predicted probability distribution of the next word (or token) and the actual word (or token).   
   2. **Formula**: For a single token prediction, the cross-entropy loss can be expressed as:Loss=−∑i=1Nyilog⁡(y^i)Loss=−i=1∑Nyilog(y^i) where yiyi is the actual distribution (one-hot encoded, where the correct token is 1 and all others are 0), and y^iy^i is the predicted probability distribution over the vocabulary.   
   3. **Training Process**:   
      * The model is trained to predict the next token in a sequence given the previous tokens. This is done over large amounts of text data.   
      * For each token in the training dataset, the model outputs a probability distribution over the vocabulary, and the cross-entropy loss is calculated using this distribution and the actual next token.   
      * The gradients of this loss with respect to the model parameters are computed, and the model parameters are updated using optimization techniques like Stochastic Gradient Descent (SGD) or its variants (e.g., Adam).   
   4. This pre-training step enables the model to learn the statistical properties of the language, which can then be fine-tuned for specific tasks such as text classification, translation, summarization, or question answering. 

   
   
   
   
   
   
   
   
What is Model Drift?   
Definition: Model drift refers to changes in a model’s predictions over time. This can be due to changes in the model’s inputs, outputs, or the actuals (ground truths).   
Importance: Monitoring for model drift is crucial because it helps maintain the accuracy and relevance of a machine learning model in a changing environment.   
   
Types of Model Drift   
Prediction Drift   
Definition: Changes in the model’s predictions over time.   
Resolution: Retrain the model with additional or updated data, or replace the model if necessary.   
   
Concept Drift   
Definition: Shift in the statistical properties of the target variable.   
Resolution: Adjust the model to account for new patterns, reweight data, or refit the model.   
   
Data Drift   
Definition: Change in the statistical properties of input data.   
Resolution: Update the model to reflect new data distributions, and continually monitor inputs.   
   
Upstream Drift   
Definition: Changes in the data pipeline feeding the model.   
Resolution: Monitor and address data quality issues, such as missing values or feature cardinality changes.   
   
   
   
   
   
   
   
Weight Initilization 

* Xavier \- sigmoid   
* He \- relu 

   
   
   
   
   
   
   
   
   
   
   
   
   
 

| Optimizer  | Description  | Learning Rate  | Memory Requirement  | Convergence Speed  | Key Features  | Pros  | Cons  |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **Gradient Descent (GD)**  | Basic optimization algorithm that updates parameters using the gradient of the loss function.  | Fixed  | Low  | Slow  | Iterates over the entire dataset.  | Simple to implement.  | Slow for large datasets.  |
| **Stochastic Gradient Descent (SGD)**  | Updates parameters using the gradient of the loss function for a single sample.  | Fixed  | Low  | Fast  | Uses one sample at a time.  | Fast, good for online learning.  | High variance in updates, may not converge smoothly.  |
| **Mini-batch Gradient Descent**  | Updates parameters using the gradient of the loss function for a small batch of samples.  | Fixed  | Low  | Moderate  | Uses batches of samples.  | Reduces variance, improves convergence.  | Requires tuning of batch size.  |
| **Momentum**  | Accelerates SGD by adding a fraction of the previous update to the current update.  | Fixed  | Moderate  | Fast  | Adds a momentum term to the update rule.  | Speeds up convergence, reduces oscillations.  | Requires tuning of momentum parameter.  |
| **Nesterov Accelerated Gradient (NAG)**  | Improves momentum by looking ahead before updating parameters.  | Fixed  | Moderate  | Fast  | Applies a correction term.  | Further reduces oscillations, faster convergence.  | Requires careful tuning.  |
| **Adagrad**  | Adapts the learning rate for each parameter individually based on past gradients.  | Decreasing  | High  | Moderate  | Accumulates squared gradients.  | Automatically adjusts learning rate, good for sparse data.  | Learning rate can become very small.  |
| **Adadelta**  | Improves Adagrad by limiting the accumulation of past gradients.  | Adaptive  | Moderate  | Moderate  | Uses a moving window of squared gradients.  | Overcomes Adagrad’s diminishing learning rate problem.  | More complex to implement.  |
| **RMSprop**  | Adapts the learning rate for each parameter using a moving average of squared gradients.  | Adaptive  | Moderate  | Fast  | Uses a moving average of squared gradients.  | Works well in practice, good for non-stationary data.  | Requires tuning of decay parameter.  |
| **Adam**  | Combines RMSprop and momentum by keeping an exponentially decaying average of past gradients and squared gradients.  | Adaptive  | High  | Fast  | Uses moving averages of gradient and squared gradient.  | Adaptive learning rate, works well in practice, good default choice.  | Requires tuning of multiple parameters.  |
| **Nadam**  | Combines Adam with Nesterov Accelerated Gradient.  | Adaptive  | High  | Fast  | Incorporates NAG into Adam.  | Benefits of Adam and NAG, faster convergence.  | Requires tuning of multiple parameters.  |
| **AdaMax**  | A variant of Adam using the infinity norm.  | Adaptive  | High  | Fast  | Uses the infinity norm for gradients.  | More stable than Adam in some cases.  | Still requires tuning of parameters.  |
| **SGD with Warm Restarts**  | Periodically restarts SGD with a large learning rate, gradually decreasing it.  | Cyclical  | Low  | Fast  | Periodically resets learning rate.  | Can escape local minima, improve generalization.  | Requires tuning of restart schedule.  |
| **Lookahead**  | Improves any optimizer by "looking ahead" at future parameter updates.  | Adaptive  | High  | Fast  | Maintains fast weights and slow weights.  | Enhances performance of base optimizer.  | Adds complexity, still requires tuning.  |

**Summary** 

* **Basic Optimizers**: Gradient Descent, SGD, and Mini-batch Gradient Descent are straightforward but may require extensive tuning and can be slow to converge.   
* **Momentum-based Optimizers**: Momentum, NAG, and Lookahead improve convergence speed and stability by incorporating previous updates.   
* **Adaptive Learning Rate Optimizers**: Adagrad, Adadelta, RMSprop, Adam, Nadam, and AdaMax adjust learning rates based on past gradients, offering faster convergence and better performance on non-stationary data.   
* **Special Techniques**: SGD with Warm Restarts can help escape local minima and improve generalization. 

   
   
   
   
   
   
   
   
   
   
   
   
Knowledge Distillation   
   
   
   
   
   
   
Hyper Parameter Optimization   
   
Manual Search, Random Search, Grid Search, and Bayesian Optimization   
   
Bayesian Optimization   
The probabilistic model is updated after each evaluation, and the next hyperparameters to evaluate are chosen based on a trade-off between exploitation (exploiting areas where the model is likely to perform well based on the surrogate model) and exploration (exploring areas where the surrogate model is uncertain or has high variance).   
   
Bayesian Optimization differs from Random Search and Grid Search in that it improves the search speed using past performances, whereas the other two methods are uniform (or independent) of past evaluations. In that sense, Bayesian Optimization is like Manual Search. Let’s say you are manually optimizing the hyperparameter of a Random Forest regression model. Firstly, you would try a set of parameters, then look at the result, change one of the parameters, rerun, and compare the results, so that way you know whether you are going towards the right direction. Bayesian Optimization does a similar thing — the performance of your past hyperparameter affects the future decision. In comparison, Random Search and Grid Search do not take into account past performance when determining new hyperparameters to evaluate. Thus, Bayesian Optimization is a much more efficient method.   
   
   
   
   
   
   
   
   
   
   
   
 

* **Hyperplane**:   
  * In SVM, a hyperplane is a decision boundary that separates different classes in the feature space. For a two-dimensional space, the hyperplane is a line; for higher dimensions, it is a flat affine subspace.   
* **Support Vectors**:   
  * Support vectors are the data points that are closest to the hyperplane and are most difficult to classify. These points determine the position and orientation of the hyperplane.   
* **Margin**:   
  * The margin is the distance between the hyperplane and the nearest support vectors. SVM aims to maximize this margin to improve the classifier's robustness and generalization.   
* **Kernel Trick**:   
  * The kernel trick allows SVM to operate in a high-dimensional space without explicitly computing the coordinates of the data in that space. Kernels transform the input data into a higher-dimensional space to make it easier to find a separating hyperplane. Common kernels include:   
    * **Linear Kernel**: Suitable for linearly separable data.   
    * **Polynomial Kernel**: Suitable for non-linear data.   
    * **Radial Basis Function (RBF) Kernel**: Popular for non-linear data and handles data well in high dimensions.   
    * **Sigmoid Kernel**: Can be used as a proxy for neural networks.   
* **C Parameter (Regularization)**:   
  * The C parameter controls the trade-off between achieving a low training error and a low testing error, which is crucial for generalization. A smaller C allows more misclassifications, leading to a wider margin, while a larger C aims for fewer misclassifications but a narrower margin.   
* **Support Vector Regression (SVR)**:   
  * SVR is the regression counterpart of SVM. It uses the same principles but aims to fit the data within a certain margin of tolerance. 

   
   
SVM Loss   
   
   
   
   
   
   
   
   
   
   
Nonparametric models are those that do not assume a fixed form for the underlying data distribution and can adapt to the complexity of the data. Unlike parametric models, which have a fixed number of parameters, nonparametric models have a number of parameters that grows with the amount of training data. Here are some examples of nonparametric models: 

1. **k-Nearest Neighbors (k-NN)**: This model makes predictions based on the k closest training examples in the feature space. It does not assume any specific functional form for the mapping from input to output.   
2. **Decision Trees**: These models recursively split the data into subsets based on feature values, creating a tree-like structure. The number of splits (and thus the complexity of the model) can grow with the size of the training data.   
3. **Kernel Density Estimation (KDE)**: This method estimates the probability density function of a random variable by averaging over a kernel function centered at each data point. The complexity of the model depends on the number of data points.   
4. **Gaussian Processes (GP)**: These models define a distribution over functions and make predictions based on the observed data. The complexity of the model increases with the number of data points.   
5. **Random Forests**: An ensemble of decision trees where each tree is trained on a random subset of the data. The number of trees and their depth can vary, making the model nonparametric.   
6. **Support Vector Machines with Non-linear Kernels**: While SVMs with linear kernels are considered parametric, using non-linear kernels (like the Radial Basis Function) can make them more flexible and capable of capturing more complex patterns in the data, behaving in a nonparametric manner in terms of their capacity to fit the data. 

   
   
   
   
   
   
   
   
   
   
   
   
**Examples of Convex Problems in ML** 

1. **Linear Regression**:   
2. **Logistic Regression**:   
3. **Support Vector Machines (SVM)**:   
4. **Ridge and Lasso Regression**: 

**Non-Convex Problems in ML** 

* **Neural Networks**   
* Clustering (e.g., k-means) 

   
   
   
   
   
   
   
   
   
   
   
**RandomForest vs Bagging**   
Both RF and Bagging use \- **Bootstrap Sampling**: 

* For each tree in the forest, a **random sample of the training data is selected with replacement**. This means that some data points can be selected multiple times, while others may not be selected at all.   
* Typically, this sample is of the same size as the original training set, but because it's sampled with replacement, some instances will be repeated, and some will be omitted. 

Feature Selection: 

* When constructing each tree, at each split, Random Forest **randomly selects a subset of features** and only considers those features for splitting the node.   
* In Bagging, all features are considered for each split in the base model. 

Model Correlation:  

* Random Forest reduces the correlation between individual models more effectively than Bagging due to the additional randomness in feature selection. 

   
   
   
   
   
   
   
μ is the mean;  σ is standard deviation;  σ^2 is the variance.   
   
   
   
   
   
Gaussian vs Normal   
Yes, the terms "Gaussian distribution" and "normal distribution" refer to the same concept in statistics and probability theory. They are two different names for the same probability distribution.   
Gaussian Distribution 

* **Definition**: Gaussian distribution is a continuous probability distribution that describes data that clusters around a mean or average.   
* **Probability Density Function (PDF)**: The PDF of a Gaussian distribution is given by: 

Normal Distribution 

* **Definition**: The normal distribution is a continuous probability distribution that is symmetric around the mean, showing that data near the mean are more frequent in occurrence than data far from the mean.   
* **Standard Normal Distribution**: A special case of the normal distribution where the mean μ is 0 and the standard deviation σ is 1\. Its PDF is: 

Key Characteristics of Gaussian/Normal Distribution 

1. **Symmetry**: The distribution is symmetric around the mean.   
2. **Bell-shaped Curve**: The shape of the PDF is a bell curve.   
3. **Mean, Median, and Mode**: All are equal and located at the center of the distribution.   
4. **68-95-99.7 Rule**: Approximately 68% of the data falls within one standard deviation of the mean, 95% within two standard deviations, and 99.7% within three standard deviations.   
5. **Central Limit Theorem**: States that the sum of a large number of independent and identically distributed (IIDs) random variables tends to follow a normal distribution, regardless of the original distribution of the variables. 

   
   
   
   
   
**Probability vs Likelihood** 

* While calculating probability, feature value can be varied, but the characteristics(mean & Standard Deviation) of the data distribution cannot be altered. 

 

* Here, the dataset features will be varied, i.e. Mean & Standard Deviation of the dataset will be varied in order to get the maximum likelihood for height \> 170 cm. 

   
   
   
   
   
   
   
   
   
   
   
   
   
   
Word2Vec loss function 

1. **Skip-gram with Negative Sampling (SGNS):** This is the most common variant. The loss function in this case is based on negative sampling, where the **objective is to distinguish the target word from a set of noise words** (negative samples) for each context word. The negative sampling loss is formulated to maximize the probability of the observed word-context pairs and minimize the probability of randomly sampled noise words.   
   The loss function LLL for a single (context, target) pair is defined as:   
2. **CBOW with Negative Sampling**: In the Continuous Bag-of-Words model, the context words are used to predict the target word. The loss function is similar to the Skip-gram model but inverted in terms of the context and target word roles.   
3. **Hierarchical Softmax:** This is an alternative to negative sampling, where a binary tree representation of the vocabulary is used. Each word is represented as a path from the root to a leaf in the tree, and the objective is to maximize the probability of the correct path. This reduces the computational complexity compared to a full softmax over the vocabulary. 

   
   
   
   
   
   
   
   
   
   
**QKV**   
QKV \- derived from the input embeddings through a learned linear transformation   
 

* **Query (Q):**   
  * The query matrix represents the set of queries for the current token or position in the input sequence.   
  * Each query vector is used to compute attention scores by measuring its similarity with key vectors from other positions.   
* **Key (K):**   
  * The key matrix represents the set of keys for all tokens in the input sequence.   
  * Each key vector helps in determining **how much attention a particular token should receive** based on its similarity to the query vectors.   
* **Value (V):**   
  * The value matrix contains the **actual values or information that are to be aggregated** or attended to.   
  * The attention mechanism uses the computed attention scores to weigh these value vectors, producing a weighted sum that represents the attended output for each position. 

   
   
   
   
   
In Transformers, the query, key, and value matrices are used in the context of attention mechanisms, particularly in models like the Transformer architecture. Here's the purpose of each: 

1. **Query Matrix (Q)**:   
   1. The query matrix is derived from the input embeddings of the model and represents what is being searched for in the input sequence.   
   2. Each query vector corresponds to a word or token in the input sequence.   
   3. During the attention computation, the query matrix is used to compute similarity scores (dot products) with the key matrix to determine how much focus to place on each token in the input sequence.   
   4. **Key Matrix (K)**:   
      1. The key matrix is also derived from the input embeddings and represents the tokens against which the similarity of the query vectors is computed.   
      2. Like the query matrix, each key vector corresponds to a word or token in the input sequence.   
      3. The key vectors are used to compute the dot product with query vectors to determine the attention weights.   
      4. **Value Matrix (V)**:   
         * The value matrix is derived from the input embeddings and represents the values or information that the model should focus on.   
         * Each value vector corresponds to a word or token in the input sequence.   
         * During the attention computation, the value vectors are weighted by the attention scores (computed from query and key) to produce the output of the attention mechanism.   
      5. In summary, these matrices (Q, K, and V) are crucial components of the self-attention mechanism in Transformers. They allow the model to compute attention scores between different tokens in the input sequence, thereby enabling the model to focus on relevant information and capture dependencies between tokens effectively. 

   
   
   
   
   
   
   
   
   
   
   
   
**Self Attention vs Cross Attention**   
Self attention \- Q is looking within the input embeddings   
Cross attention \- Q is taking from output embeddings   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
Maximum Likelihood Estimate   
Probabilistic framework for solving the problem of density estimation \-- finding mean and standard deviation of PDF   
   
 

* It involves maximizing a likelihood function in order to find the probability distribution and **parameters that best explain** the observed data.   
* It provides a framework for predictive modeling in machine learning **where finding model parameters can be framed as an optimization problem.**   
* Used in K-Means 

   
   
   
   
   
   
   
   
   
   
   
   
Bernoulli's trial is also said to be a binomial trial. In the case of the Bernoulli trial, there are only two possible outcomes. But, in the case of the binomial distribution, we get the number of successes in a sequence of independent experiments.   
   
   
   
   
   
   
   
   
   
   
MAP vs MLE 

* MAP and MLE are both methods used to estimate parameters in statistical models, but they differ in their approach to incorporating prior information and their interpretation.    
* **MLE** is a frequentist approach that maximizes the likelihood of the data given the parameters, assuming no prior information.   
* **MAP** is a Bayesian approach that incorporates prior beliefs about the parameters, balancing them with the likelihood of the observed data to estimate the parameters. 

   
   
   
   
   
   
   
   
Expectation Maximization   
 

* Given the number of guassinas, find their parameters (appropriate mean and variance). Initialise the parameters with random values.   
* EM doesnt assign a class but instead computes a probability for a certain point to be of a certain class.   
* **K-means does almost same but does hard classificaiton (binary), but EM does soft clustering \-- computes probability**   
* Uses probabilities to readjust mean and variance exactly like K-means 

   
   
   
   
   
MLE vs EM 

* MLE accumulates all of the data first and then uses that data to construct the most likely model. EM takes a guess at the parameters first—accounting for the missing data—then tweaks the model to fit the guesses and the observed data.   
* MLE computes the density of fitting one distribution to the data; EM Fits various gaussians without knowing their classes 

   
   
   
   
   
   
   
   
What is non-linearity give some examples.   
Linearity means homogeneity of degree 1 and additiveness. This means, given a function 𝑓(𝑥), it should be both: 

1. homogeneous of degree 1, which means,𝑓(𝑎𝑥)=𝑎𝑓(𝑥)   
2. Additive, which means 𝑓(𝑥+𝑦)=𝑓(𝑥)+𝑓(𝑦) 

   
And intuitively, any function that doesn’t abide by the above properties are considered as ‘non-linear’. In machine learning, commonly-used non-linear functions include feature maps like polynomial, anova and non-linear activations in Neural Networks like RELU and Sigmoid.   
One might wonder why we need non-linearity. From the perspective of matrix analysis, linear transformations are essentially only **projecting vectors to another base**. This brings **limited expressivity** of linear functions. For example, for a multi-layer fully-connected Neural Network, given the optimal projection existed, adding layers could not help in improving the results (although it might help in faster convergence). (There is a strict proof about this, but I’m not going to include it here.) Another example is for regression problems, when the true map between x and y are non-linear, linear regression could not provide a satisfying result thus we will be needing feature maps/ kernel tricks.   
   
   
   
   
   
Desirable properties of activation functions   
Non Linearity (Non Constant) 

* The purpose of the activation function is to introduce non-linearity into the network in turn allows you to model a response variable (aka target variable, class label, or score) that varies non-linearly with its explanatory variables   
* Non-linear means that the output cannot be reproduced from a linear combination of the inputs   
* Another way to think of it: without a non-linear activation function in the network, a NN, no matter how many layers it had, would behave just like a single-layer perceptron, because summing these layers would give you just another linear function (see definition just above). 

Continuous (differentiable) 

* This property is necessary for enabling gradient-based optimization methods.   
* The binary step activation function is not differentiable at 0, and it differentiates to 0 for all other values, so gradient-based methods can make no progress with it 

Range 

* When the range of the activation function is finite, gradient-based training methods tend to be more stable, because pattern presentations significantly affect only limited weights.   
* When the range is infinite, training is generally more efficient because pattern presentations significantly affect most of the weights. In the latter case, smaller learning rates are typically necessary. 

Monotonic (non-decreasing or non-increasing) 

* When the activation function is monotonic, the error surface associated with a single-layer model is guaranteed to be convex. 

Approximates identity near the origin 

* When activation functions have this property, the neural network will learn efficiently when its weights are initialized with small random values.   
* When the activation function does not approximate identity near the origin, special care must be used when initializing the weights.   
* Output of the activation function should be symmetrical at zero so that the gradients do not shift to a particular direction. 

   
   
   
   
   
Why do we use activation functions   
Allows non linearity, enabling to approximate a function.   
   
   
   
   
   
   
   
Gradient Saturation   
If you use sigmoid-like activation functions, like sigmoid and tanh, after some epochs of training, the linear part of each neuron will have values that are very big or very small. This means that the linear part will have a big output value regardless of its sign. Consequently, the input of sigmoid-like functions in each neuron which adds non-linearity will be far from the center of these functions.   
   
In those locations, the gradient/derivative value is very small. Consequently, after numerous iterations, the weights get updated so slowly because the value of the gradient is very small. This is why we use the ReLU activation function for which its gradient doesn't have this problem.   
   
The biggest advantage of ReLu is indeed non-saturation of its gradient, which greatly accelerates the convergence of stochastic gradient descent compared to the sigmoid / tanh functions.   
   
   
   
   
   
   
   
   
Batch Gradient Descent vs Stochastic Gradient Descent   
   
   
   
   
   
   
   
   
   
   
Gradient Descent Derivation   
   
   
   
   
   
   
   
   
Bias Variance Tradeoff   
   
   
   
   
The bias-variance trade-off is a fundamental concept in supervised machine learning that deals with the balance between model simplicity and model flexibility: 

1. **Bias**: Bias refers to the error introduced by approximating a real-world problem with a simplified model. A high bias model oversimplifies the underlying relationships in the data and may consistently underfit, leading to poor performance on both training and test data.   
2. **Variance**: Variance, on the other hand, refers to the model's sensitivity to small fluctuations in the training data. A high variance model is overly complex, capturing noise in the training data rather than the underlying relationships, which can lead to overfitting—performing well on training data but poorly on unseen test data. 

   
**Trade-off Explanation:** 

* **High Bias, Low Variance (Underfitting)**:   
  * Models with high bias and low variance are too simplistic to capture the underlying patterns in the data.   
  * They tend to underfit, performing poorly both on the training data and on unseen test data.   
  * Examples include linear models that cannot capture nonlinear relationships in the data.   
* **Low Bias, High Variance (Overfitting)**:   
  * Models with low bias and high variance are overly complex, fitting the training data too closely.   
  * They capture noise and fluctuations in the training data, which do not generalize well to new, unseen data.   
  * Examples include decision trees with no pruning or high-degree polynomial regression models. 

   
 

* **Overfitting**: Model is too complex, captures noise, performs well on training data but poorly on testing data. Solutions include simplifying the model, regularization, and obtaining more data.   
* **Underfitting**: Model is too simple, fails to capture underlying patterns, performs poorly on both training and testing data. Solutions include using a more complex model, feature engineering, and reducing regularization.   
* low bias and high variance \--- overfit: the model will learn "by heart", close to the training data (generalise too little). Can be noticed with: low error on the training but a high error on test/validation.   
* high bias and low variance \--- underfit: the model will generalize too much and possibly miss patterns/trends... 

   
   
**Reduce Bias (Underfitting):** 

1. **Increase Model Complexity**:   
2. **Feature Engineering**:   
3. **Reduce Regularization**:   
4. **Increase Training Time**: 

**Reduce Variance (Overfitting):** 

1. **Regularization**:   
2. **Cross-Validation**:   
3. **Feature Selection**:   
4. **Ensemble Methods**:   
5. **Early Stopping**:   
6. **Data Augmentation**: 

   
   
   
   
   
   
   
L1 Backpropagation 

* Differentiation when L1 is not differentiable   
* L1 norm is based on the absolute value of the difference, and absolute value |x| has a kink at x=0. It is not differentiable at the kink.   
* The absolute value (or the modulus function), i.e. 𝑓(𝑥)=|𝑥| is not differentiable is the way of saying that its derivative is not defined for its whole domain. For modulus function the derivative at 𝑥\=0 is undefined. 

   
𝑑|𝑥|/𝑑𝑥 \= −1 if 𝑥\<0 and 1 if 𝑥\>0 (not defined when x=0)   
   
L1 (lasso) vs L2 (ridge) 

* L1 is absolute norm which can take coefficient to zero and can act as automatic feature selection.   
* L1 corresponds to setting a Laplacean prior on the terms, while L2 corresponds to a Gaussian prior.   
* L1 is used with laplacian prior because we assume that most of the features are not needed due feature scaling.   
* L2 is square norm which keep every coefficient above zero and minimize slope and intercept.   
* L2 regularization disperse the error terms in all the weights that leads to more accurate customized final models. 

   
   
   
   
   
   
   
   
   
Early stopping   
stop the training early once you start seeing the drop in the accuracy.   
   
   
Gradient Descent   
   
IID Independent and Identically Distributed 

* Thus, we say that random variables X₁, X₂, …, Xn are all independent and identically distributed if all Xᵢ are mutually independent and they all have (or belong to) the same distribution.   
* Suppose I flip a fair coin 100 times and get head 53 times and tail 47 times. And, now I want to flip my coin again for the 101st time. What will I get? Well, I can either get a head or a tail with a probability of 0.5 for each (since it is a fair coin). However, the probability of getting a head or a tail does not depend on any of the previous outcomes 

   
   
   
   
   
   
   
**Metrics**   
[https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/](https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/)   
[ROC and AUC, Clearly Explained\!](https://www.youtube.com/watch?app=desktop&v=4jRBRDbJemM)   
   
Precision: Out of all the times the model said positive, how many were really positive.   
   
Recall: Out of the actual positives, how many were classified correctly.   
   
F1: Useful when there is class imbalance. Its the harmonic mean of the precision and recall of a classification model. The two metrics contribute equally to the score, ensuring that the F1 metric correctly indicates the reliability of a model.   
   
   
   
   
 

* If labels are more or less equally sized (have roughly the same number of instances), use any.   
* If there are labels with more instances than others and if you want to bias your metric towards the **most** populated ones, use **micr**.   
* If there are labels with more instances than others and if you want to bias your metric toward the **least** populated ones (or at least you don't want to bias toward the most populated ones), use **macr**. 

   
   
   
   
   
False positive rate \= 1 \- Specificity   
True positive rate \= TP/P \= recall \= sensitivity   
   
ROC \---- TPR vs FPR   
   
   
   
   
   
   
   
Confusion Matrix 

* **Type 1** error is when your algorithm makes a positive prediction, but in fact, it’s negative. For example, your algorithm predicted a patient has cancer, but in fact, he doesn’t.   
* **Type 2** error is when your algorithm makes a negative prediction, but in fact, it’s positive. For example, your algorithm predicted a patient doesn’t have cancer, but in fact, they do. 

   
   
   
   
   
   
   
   
Loss Functions   
   
[https://vinija.ai/concepts/loss/](https://vinija.ai/concepts/loss/)   
   
   
   
   
   
   
   
Triplet Loss   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
Why is “Naive” Bayes naive? 

* Despite its practical applications, especially in text mining, Naive Bayes is considered “Naive” because it makes an assumption that is virtually impossible to see in real-life data: the conditional probability is calculated as the pure product of the individual probabilities of components. This implies the absolute independence of features — a condition probably never met in real life.   
* computes the posterior probability of an event given what is known as prior knowledge. 

   
   
Batch normalization   
The technique to improve the performance and stability of neural networks by normalizing the inputs in every layer so that they have mean output activation of zero and standard deviation of one.   
   
   
   
   
   
   
   
Covariance vs Correlation   
Covariance is when two variables vary with each other, whereas Correlation is when the change in one variable results in the change in another variable.   
The value of covariance between 2 variables is achieved by taking the summation of the product of the differences from the means of the variables as follows:    
   
Covariance shows you how the two variables differ, whereas correlation shows you how the two variables are related.   
The main result of a correlation is called the correlation coefficient.   
   
   
Why correlation is preferred over covariance?   
because it remains unaffected by the change in location and scale, and can also be used to make a comparison between two pairs of variables.   
   
   
   
   
   
Multi Layer Perceptron (NN)   
MLP is fully connected feed-forward network. In particular CNN which is partially connected, RNN which has feedback loop are not MLPs. Multi-Layer Perceptron is a model of neural networks (NN). There are several other models including recurrent NN and radial basis networks.   
   
   
   
   
   
   
   
Is a fully connected neural network equal to a feed-forward neural network?   
Feed forward architecture implies absence of recurrent or feedback connections. The path is only forward facing, no backward feed connections between neurons are present. Thus the signal can only be fed forward hence the name feed-forward neural network (NN).   
 

* A [Feed-Forward Neural Network](http://en.wikipedia.org/wiki/Feedforward_neural_network) is a type of Neural Network architecture where the connections are "fed forward", i.e. do not form cycles (like in recurrent nets).   
* The term "Feed forward" is also used when you input something at the input layer and it travels from input to hidden and from hidden to output layer. The values are "fed forward". 

   
There are many types of feed forward NNs such as the: 

* Fully connected multi layer neural networks such as the multi-layer perceptrons (MLP).   
* Fully Convolutional neural networks.   
* Convolutional neural networks \+ fully connected layers (normally just called convolutional neural networks) 

   
There is another group called recurrent neural networks (RNN) with recurrent or feedback connections between neurons. Such networks are turing complete, in the sense that they can learn any function, spatial \+ temporal functions. 

* Vanilla RNN which suffers mostly from vanishing/exploding gradient problem.   
* Long short term memory (LSTM) networks.   
* Gated recurrent unit (GRU) networks.   
* Recurrent convolutional neural networks (RCNN). 

   
Another varient are the so called memory augmented neural networks or memory networks in short. Which as the name implies are about a memory block hooked up to a neural network. They can somehow learn to reason. 

* Neural turing machines (NTM)   
* Differentiable neural computers (DNC). 

   
   
   
   
   
Backpropagation vs Feed Forward 

* Backpropagation is algorithm to train (adjust weight) of neural network. Input for backpropagation is output\_vector, target\_output\_vector, output is adjusted\_weight\_vector.   
* Feed-forward is algorithm to calculate output vector from input vector. Input for feed-forward is input\_vector, output is output\_vector.   
* When you are training neural network, you need to use both algorithms.   
* When you are using neural network (which have been trained), you are using only feed-forward.   
* Basic type of neural network is multi-layer perceptron, which is Feed-forward backpropagation neural network. 

   
   
   
   
   
   
   
   
Vanishing and Exploding Gradients 

* it means the gradients are so small that the learning will be very slow.   
* The reason for vanishing gradient is that during backpropagation, the gradient of early layers (layers near to the input layer) are obtained by multiplying the gradients of later layers (layers near to the output layer). So, for example if the gradients of later layers are less than one, their multiplication vanishes very fast.   
* RNNs suffer from the problem of vanishing gradients, which hampers learning of long data sequences.   
* LSTM was invented specifically to avoid the vanishing gradient problem.   
* Model with exploding gradients produce NaN 

**Fix** 

* Weight regularization   
* use LSTM   
* use Gradient Clipping (limit gradients to a certain threshold for exploding gradients)   
   

   
   
   
   
   
   
   
   
   
RNN vs LSTM vs GRU   
   
   
   
**Recurrent Neural Networks (RNNs)**   
**Pros**: 

* **Simple Architecture**: RNNs have a straightforward architecture, making them easy to implement and understand.   
* **Efficient for Short Sequences**: They work well for tasks involving short sequences where long-term dependencies are not critical. 

**Cons**: 

* **Vanishing Gradient Problem**: RNNs suffer from the vanishing gradient problem, making it difficult to learn long-term dependencies.   
* **Limited Memory**: They struggle to retain information from far in the past, limiting their effectiveness on longer sequences.   
* **Training Instability**: Training can be unstable and slow due to gradient issues. 

**Long Short-Term Memory Networks (LSTMs)** 

* **Forget Gate**:   
  * Decides what portion of the previous cell state to forget.   
* **Input Gate**:   
  * Decides what new information to store in the cell state.   
* **Output Gate**:   
  * Decides what part of the cell state to output. 

**Pros**: 

* **Long-Term Dependencies**: LSTMs are designed to capture long-term dependencies in the data, thanks to their memory cell and gating mechanisms.   
* **Mitigate Vanishing Gradient**: The architecture of LSTMs helps to mitigate the vanishing gradient problem, allowing for more stable and effective training on longer sequences.   
* **Versatile**: They perform well on a variety of sequential tasks, including time series prediction, text generation, and speech recognition. 

**Cons**: 

* **Complex Architecture**: LSTMs have a more complex architecture compared to simple RNNs, making them harder to implement and understand.   
* **Computationally Intensive**: They require more computational resources and memory due to the additional gates and parameters.   
* **Longer Training Times**: The complexity of the architecture can lead to longer training times. 

**Gated Recurrent Units (GRUs)** 

* **Reset Gate**:   
  * Controls how much of the previous hidden state to forget.   
* **Update Gate**:   
  * Decides how much of the previous hidden state and the current new candidate state to combine to update the hidden state. 

**Pros**: 

* **Simpler Architecture**: GRUs have a simpler architecture compared to LSTMs, with fewer gates (reset and update gates) and parameters, making them easier to implement and faster to train.   
* **Efficient Training**: They often train faster and require less computational power and memory compared to LSTMs.   
* **Handle Long-Term Dependencies**: Like LSTMs, GRUs are designed to handle long-term dependencies and can mitigate the vanishing gradient problem.   
* **Performance**: GRUs often perform comparably to LSTMs on a variety of tasks, sometimes even outperforming them, especially when computational efficiency is a concern. 

**Cons**: 

* **Less Interpretability**: The simpler gating mechanism may not capture all aspects of long-term dependencies as effectively as LSTMs in certain cases.   
* **Task Specificity**: While GRUs work well on many tasks, there are cases where the additional complexity of LSTMs provides a performance advantage. 

**Summary** 

* **RNNs**: Best for short sequences and simpler tasks due to their simplicity, but they struggle with long-term dependencies and suffer from the vanishing gradient problem.   
* **LSTMs**: Excellent for tasks requiring long-term memory and handling long sequences, but they are computationally intensive and have a complex architecture.   
* **GRUs**: Offer a balance between RNNs and LSTMs, with a simpler architecture and faster training times while still handling long-term dependencies effectively. They are often preferred when computational efficiency is important. 

   
   
   
   
    
    
**Can we use RNN for image processing**   
Pixel RNN   
   
   
   
   
   
   
   
   
Normlization   
Normalization is a scaling technique in which values are shifted and rescaled so that they end up **ranging between 0 and 1**. It is also known as Min-Max scaling.   
   
   
Standardization   
Standardization is another scaling technique where the **values are centered around the mean with a unit standard deviation**. This means that the mean of the attribute becomes zero and the resultant distribution has a unit standard deviation.   
   
Why Feature Scaling 

* It is required only when features have different ranges.   
* Having features on a similar scale can help the gradient descent converge more quickly towards the minima.   
* In distance based algorithm, the higher values overpower the lower valued features   
* The model weights shoot up and will not learn from raw data.   
* Computational benefits. 

   
Normalize or Standardize? 

* Normalization is good to use when you know that the **distribution of your data does not follow a Gaussian distribution**. This can be useful in algorithms that do not assume any distribution of the data like K-Nearest Neighbors and Neural Networks.   
* Normalization is good to use when you know that the distribution of your data does not follow a Gaussian distribution. This can be useful in algorithms that do not assume any distribution of the data like K-Nearest Neighbors and Neural Networks.   
* Standardization, on the other hand, can be helpful in cases where the **data follows a Gaussian distribution**. However, this does not have to be necessarily true. Also, unlike normalization, standardization does not have a bounding range. So, even if you have outliers in your data, they will not be affected by standardization. 

   
   
   
[https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/](https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/)   
   
   
   
   
   
Batch Normalisation 

* A typical neural network is trained using a collected set of input data called batch. Similarly, the normalizing process in batch normalization takes place in batches, not as a single input.   
* h is the input to the hidden layer, so we apply normalisation at a particular layer before passing through the activation. 

   
   
   
   
Restricted Boltzmann machine   
RBMs are a variant of [Boltzmann machines](https://en.wikipedia.org/wiki/Boltzmann_machine), with the restriction that their [neurons](https://en.wikipedia.org/wiki/Artificial_neural_network#Neuron) must form a [bipartite graph](https://en.wikipedia.org/wiki/Bipartite_graph): a pair of nodes from each of the two groups of units (commonly referred to as the "visible" and "hidden" units respectively) may have a symmetric connection between them; and there are no connections between nodes within a group. By contrast, "unrestricted" Boltzmann machines may have connections between [hidden units](https://en.wikipedia.org/wiki/Artificial_neural_network#Hidden_units). This restriction allows for more efficient training algorithms than are available for the general class of Boltzmann machines, in particular the [gradient-based](https://en.wikipedia.org/wiki/Gradient_descent) contrastive divergence algorithm.   
   
   
   
   
   
   
   
What’s the difference between KNN and K-means?   
KNN is a supervised model used for classification   
K-Means is an unsupervised model used for clustering.   
   
   
   
   
   
   
Cross Validation   
K-Fold 

* Split into k folds and train k-1 models leaving one fold as test set   
* During inference, uses all the k-1 models 

Leave one out 

* k \= n (data points)   
* only one test point as test set   
* suitable for small datasets 

   
   
   
   
   
Pruning a decision tree   
Pruning reduces the size of decision trees by removing parts of the tree that do not provide power to classify instances.   
   
Overfitting in Decision Trees 

* Decision trees are prone to overfitting, especially when a tree is particularly deep.   
* They can form a branch for each data point and wont generalise 

   
   
   
   
   
   
   
   
   
   
   
Cross entropy   
   
Logistic regression uses   
Log Loss=−(ylog(y^)+(1−y)log(1−y^))   
   
   
Curse of Dimensionality 

* If we have more features than observations than we run the risk of massively overfitting our model — this would generally result in terrible out of sample performance.   
* PCA is used to reduce the dimensitions   
* A typical rule of thumb is that there should be at least 5 training examples for each dimension in the representation.   
* This phenomenon states that with a fixed number of training samples, the average (expected) predictive power of a classifier or regressor first increases as the number of dimensions or features used is increased but beyond a certain dimensionality it starts deteriorating instead of improving steadily. 

   
   
Eigenvalues and Eigenvector   
In [linear algebra](https://en.wikipedia.org/wiki/Linear_algebra), an **eigenvector** of a [linear transformation](https://en.wikipedia.org/wiki/Linear_map) is a nonzero [vector](https://en.wikipedia.org/wiki/Vector_space) that changes at most by a [scalar](https://en.wikipedia.org/wiki/Scalar_\(mathematics\)) factor when that linear transformation is applied to it. The corresponding **eigenvalue**, often denoted by is the factor by which the eigenvector is scaled.   
   
AX=λX   
   
   	(𝐴−𝜆𝐼)𝑋\=0   
   
λ is called an eigenvalue   
   
   
eigen values: λ1=-1, λ2=-2   
   
   
eigen vector for λ1   
   
   
   
   
eigen vector for λ2   
   
   
   
   
   
PCA vs SVD 

* **Connection**: PCA can be viewed as a specific application of SVD where the data matrix is centered (mean-subtracted) before applying SVD on its covariance matrix.   
* **Dimensionality Reduction**: While both techniques can be used for dimensionality reduction, PCA explicitly focuses on identifying axes that capture maximum variance, whereas SVD provides a decomposition that can be used for various purposes beyond dimensionality reduction.   
* **Data Handling**: PCA is typically used on data matrices, whereas SVD is applied to any matrix, including rectangular and sparse matrices. 

   
   
   
   
   
   
   
Model Calibration   
Model calibration can be defined as finding a unique set of model parameters that provide a good description of the system behaviour, and can be achieved by confronting model predictions with actual measurements performed on the system.   
   
   
Exploration vs Exploitation strategies   
The exploration-exploitation trade-off is a well-known problem that occurs in scenarios where a learning system has to repeatedly make a choice with uncertain pay-offs. In essence, the dilemma for a decision-making system that only has incomplete knowledge of the world is whether to repeat decisions that have worked well so far (exploit) or to make novel decisions, hoping to gain even greater rewards (explore).   
   
ϵ-greedy 

* ϵ-greedy lets you decide what fraction of your decisions you want to spend exploring (ϵ) and what fraction you want to spend exploiting (1-ϵ) the best option so far. 

   
Upper Confidence Bound (UCB) 

* At every step, we don’t just pick the option with the highest estimated mean reward μ, as we would with the greedy approach, but we instead go for the option with the highest μ plus one standard deviation σ.   
* We call μ \+ βσ the [upper confidence bound](http://jmlr.org/papers/volume3/auer02a/auer02a.pdf), where beta is a trade-off parameter, used to steer towards more or less exploration. As β tends to zero, we get closer to pure exploitation of the option that provided the highest mean reward thus far. High β values, on the other hand, favor exploring until we have little uncertainty left. 

   
Thompson Sampling 

* At every time-step, we draw one sample from each distribution. We then rank the options based on the reward-value of these samples, just as we ranked the distributions based on mean plus β times the standard deviation before. Finally, we pick the highest-ranked option. 

   
   
Bagging vs Boosting   
Bagging \- Combines the predictions from multiple machine learning algorithms together to make more accurate predictions than any individual model. Used to reduce the variance for those algorithm that have high variance.   
Boosting \- an [ensemble](https://en.wikipedia.org/wiki/Ensemble_learning) [meta-algorithm](https://en.wikipedia.org/wiki/Meta-algorithm) for primarily reducing [bias](https://en.wikipedia.org/wiki/Supervised_learning#Bias-variance_tradeoff), and also variance in [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning), and a family of machine learning algorithms that convert weak learners to strong ones. Takes several weak models, aggregating the predictions to select the best prediction.   
 

* Very roughly, we can say that bagging will mainly focus at getting an ensemble model with less variance than its components   
* boosting and stacking will mainly try to produce strong models less biased than their components (even if variance can also be reduced). 

   
   
   
   
Decision Tree vs Random Forest 

* When using a decision tree model on a given training dataset the accuracy keeps improving with more and more splits. You can easily overfit the data and doesn't know when you have crossed the line unless you are using cross validation (on training data set). The advantage of a simple decision tree is model is easy to interpret, you know what variable and what value of that variable is used to split the data and predict outcome.   
* A random forest is like a black box and works as mentioned in above answer. It's a forest you can build and control. You can specify the number of trees you want in your forest(n\_estimators) and also you can specify max num of features to be used in each tree. But you cannot control the randomness, you cannot control which feature is part of which tree in the forest, you cannot control which data point is part of which tree. Accuracy keeps increasing as you increase the number of trees, but becomes constant at certain point. Unlike decision tree, it won't create highly biased model and reduces the variance. 

When to use a decision tree: 

1. When you want your model to be simple and explainable    
2. When you want non parametric model    
3. When you don't want to worry about feature selection or regularization or worry about multi-collinearity.    
4. You can overfit the tree and build a model if you are sure of validation or test data set is going to be subset of training data set or almost overlapping instead of unexpected. 

When to use random forest : 

1. When you don't bother much about interpreting the model but want better accuracy.    
2. Random forest will reduce variance part of error rather than bias part, so on a given training data set decision tree may be more accurate than a random forest. But on an unexpected validation data set, Random forest always wins in terms of accuracy. 

   
Non Parametric Model   
A nonparametric model is one which cannot be parametrized by a fixed number of parameters 

* k-Nearest Neighbors   
* Decision Trees like CART and C4.5   
* Support Vector Machines (when without parameters) 

   
   
RNN vs LSTM vs GRU   
   
   
 

* GRU is related to LSTM as both are utilizing different way if gating information to prevent vanishing gradient problem.   
* The GRU controls the flow of information like the LSTM unit, but without having to use a ***memory unit***. It just exposes the full hidden content without any control.   
* GRU performance is on par with LSTM, but computationally ***more efficient*** (*less complex structure as pointed out*).   
* GRUs train faster and perform better than LSTMs on less training data if you are doing language modeling (not sure about other tasks).    
* GRUs are simpler and thus easier to modify, for example adding new gates in case of additional input to the network. It's just less code in general.   
* LSTMs should in theory remember longer sequences than GRUs and outperform them in tasks requiring modeling long-distance relations. 

   
   
Bayesian Optimization 

* There are 4 types of hyperparameter optimization: Manual Search, Random Search, Grid Search, and Bayesian Optimization   
* In Bayesian, The performance of your past hyperparameter affects the future decision.   
* In comparison, Random Search and Grid Search do not take into account past performance when determining new hyperparameters to evaluate. Thus, Bayesian Optimization is a much more efficient method.   
* Say we want to find a set of hyperparameters that will minimize RMSE. Here, the function to compute RMSE is called objective function. If we were to know the probability distribution of our objective function, (in simple words, if we were to know the shape of the objective function), then we can simply compute the gradient descent and find the global minimum.   
* **Bayesian Optimization builds a probability model of the objective function and uses it to select hyperparameter to evaluate in the true objective function.**   
* we need to build a surrogate model (also called response surface model) to approximate the true objective function.   
* A surrogate model by definition is “the probability representation of the objective function”, which is essentially a model trained on the (hyperparameter, true objective function score) pairs. In math, it is p(objective function score | hyperparameter). 

SMBO stands for Sequential Model-Based Optimization, which is another name of Bayesian Optimization. It is “sequential” because the hyperparameters are added to update the surrogate model one by one; it is “model-based” because it approximates the true objective function with a surrogate model that is cheaper to evaluate.   
   
   
Why Gaussian radial basis function maps the examples into an infinite-dimensional space? 

* If you have m distinct training points then the gaussian radial basis kernel makes the SVM operate in an m dimensional space. We say that the radial basis kernel maps to a space of infinite dimension because you can make m as large as you want and the space it operates in keeps growing without bound.   
* However, other kernels, like the polynomial kernel do not have this property of the dimensionality scaling with the number of training samples. For example, if you have 1000 2D training samples and you use a polynomial kernel of \<x,y\>^2 then the SVM will operate in a 3 dimensional space, not a 1000 dimensional space. 

   
Types of Variables   
Dependent and Independent Variables 

* An independent variable, sometimes called an experimental or predictor variable, is a variable that is being manipulated in an experiment in order to observe the effect on a dependent variable, sometimes called an outcome variable.   
* The dependent variable is simply that, a variable that is dependent on an independent variable(s).  

Categorical 

| Nominal variables  | that have two or more categories, but which do not have an intrinsic order.   |
| :---- | :---- |
| Dichotomous variables  | nominal variables which have only two categories or levels.  |
| Ordinal variables   | that have two or more categories just like nominal variables only the categories can also be ordered or ranked.  |

Continuous 

| Interval variables  | Variables for which their central characteristic is that they can be measured along a continuum and they have a numerical value.  |
| :---- | :---- |
| Ratio variables  | Are interval variables, but with the added condition that 0 (zero) of the measurement indicates that there is none of that variable.   |

   
   
Train an Ordinal Regression with any Classifier 

* binary target is 1 if ratings \> 1, so the classifier will predict Pr(Target \> 1\)   
* binary target is 1 if ratings \> 2, so the classifier will predict Pr(Target \> 2\)   
* binary target is 1 if ratings \> 4, so the classifier will predict Pr(Target \> 3\)   
* binary target is 1 if ratings \> 4, so the classifier will predict Pr(Target \> 4\) 

   
   
Pr(y=1) \= 1-Pr(Target \> 1\)   
Pr(y=2) \= Pr(Target\>1)-P(Target \> 2\)   
Pr(y=3) \= Pr(Target\>2)-P(Target \> 3\)   
Pr(y=4) \= Pr(Target\>3)-P(Target \> 4\)   
Pr(y=5) \= Pr(Target\>4)   
   
Cross Entropy   
   
Adam vs AdaGrad vs RMSProp   
[https://ruder.io/optimizing-gradient-descent/](https://ruder.io/optimizing-gradient-descent/)   
   
   
Activation Functions   
   
[https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)   
   
   
Batch norm 

* The Batch Normalization Layer is actually inserted right after a Conv Layer/Fully Connected Layer, but before feeding into ReLu (or any other kinds of) activation.   
* Dropout is applied after activation layer.   
* CONV/FC \-\> BatchNorm \-\> ReLu(or other activation) \-\> Dropout \-\> CONV/FC 

 

1. Faster Training   
2. Improves Regularization   
3. Improves Accuracy 

   
   
Naive Bayes   
Multinomial Naive Bayes calculates likelihood to be count of an word/token (random variable) and Gaussian Naive Bayes calculates likelihood to be following:   
   
   
   
Dropout   
Dropout is a regularization method that approximates training a large number of neural networks with different architectures in parallel.   
   
   
CBOW vs Skipgram   
Word2Vec has two types: CBOW and SkipGram 

* As we know, **CBOW is learning to** **predict the word** by the context. Or maximize the probability of the target word by looking at the context. And this happens to be a problem for rare words. For example, given the context yesterday was a really \[...\] day CBOW model will tell you that most probably the word is beautiful or nice. Words like delightful will get much less attention of the model, because it is designed to predict the most probable word. This word will be smoothed over a lot of examples with more frequent words.   
* On the other hand, the **skip-gram model is designed to predict the context**. Given the word delightful it must understand it and tell us that there is a huge probability that the context is yesterday was really \[...\] day, or some other relevant context. With skip-gram the word delightful will not try to compete with the word beautiful but instead, delightful+context pairs will be treated as new observations. 

   
Skip-gram: works well with small amount of the training data, represents well even rare words or phrases.   
CBOW: several times faster to train than the skip-gram, slightly better accuracy for the frequent words   
   
   
Glove vs Fasttext 

* Glove treats each word in corpus like an atomic entity and generates a vector for each word. In this sense Glove is very much like word2vec- both treat words as the smallest unit to train on.   
* Fasttext which is essentially an extension of word2vec model, treats each word as composed of character ngrams. So the vector for a word is made of the sum of this character n grams. This difference enables fasttext to 

   
 

1. In ***word2vec, Skipgram*** models try to capture co-occurrence one window at a time   
2. In ***Glove ***it tries to capture the counts of overall statistics how often it appears. 

   
   
SVD   
   
M=UΣVᵗ, where:   
M-is original matrix we want to decompose   
U-is left singular matrix (columns are left singular vectors). U columns contain eigenvectors of matrix MMᵗ   
Σ-is a diagonal matrix containing singular (eigen) values   
V-is right singular matrix (columns are right singular vectors). V columns contain eigenvectors of matrix MᵗM   
   
   
SVD is more general than PCA. From the previous picture we see that SVD can handle matrices with different number of columns and rows. SVD is similar to PCA. PCA formula is M=𝑄𝚲𝑄ᵗ, which decomposes matrix into orthogonal matrix 𝑄 and diagonal matrix 𝚲. Simply this could be interpreted as:   
   
   
   
   
   
   
   
Batch norm   
Optimisers   
Activation   
NB   
Dropout   
Bagging boosting   
Overfitting and Underfitting   
SVM   
Bias and Variance   
   
   
   
   
   
   
Supervised learning:   
Topics: linear and logistic regression, Random Forests, Gradient Boosted trees, k-nearest neighbor methods, Naïve Bayes, kernel methods, Stochastic Gradient Descent (SGD), Bayesian linear regression, Gaussian Processes, concepts of overfitting and under fitting, regularization, and evaluation metrics for classification and regression problems.   
   
Deep neural networks:   
Topics: feedforward neural networks (NNs), Convolutional Neural Networks (CNNs), backpropagation, Recurrent Neural Networks (RNNs), and Long Short Term Memory (LSTM) networks.     
   
Unsupervised learning:   
Topics: clustering algorithms, Expectation Maximization (EM), Gaussian Mixture Models, and evaluation metrics for clustering problems.   
   
Probabilistic Graphical models:   
Topics: topic models such as Latent Dirichlet Allocation (LDA) and inference methods such as Belief Propagation, Gibbs Sampling, and Variational Inference.   
   
Dimensionality reduction:   
Topics: Principal Component Analysis (PCA), Singular Value Decomposition (SVD), Spectral Clustering and Matrix Factorization.   
   
Sequential models:   
Topics: Conditional Random Fields (CRFs), Hidden Markov Models (HMMs), and Natural Language processing applications such as Named Entity Recognition (NER) and Parts of Speech (POS) tagging.   
   
Reinforcement learning:   
Topics: explore-exploit techniques, multi-armed bandits (epsilon greedy, UCB, Thompson Sampling), Q-learning, and Deep Q-Networks (DQNs).   
   
   
   
   
Expect to be asked about data-driven modeling, train/test protocols, error analysis, and statistical significance   
   
   
Matrix factorization methods   
SVD and eigen decomposition   
XGboost and GBM   
   
   
Propensity Model and how beta estimates are calculated by MLE?   
Time Series Model and meaning and calculation of ACF and PACF?   
how to calculate sample size?     
Conjugate priors in Gibbs sampling   
Collapsed Gibbs sampling   
Categorical vs ordinal   
Distributions for categorical and real   
Is Deep learning better than ML for normal text classification tasks.   
   
   
   
   
   
[https://vinija.ai/concepts/](https://vinija.ai/concepts/)   
   
   
   
   
   
   
[https://www.geeksforgeeks.org/find-number-of-islands/](https://www.geeksforgeeks.org/find-number-of-islands/)   
   
[https://www.geeksforgeeks.org/longest-increasing-subsequence-dp-3/](https://www.geeksforgeeks.org/longest-increasing-subsequence-dp-3/)   
   
[https://www.geeksforgeeks.org/sort-array-wave-form-2/](https://www.geeksforgeeks.org/sort-array-wave-form-2/)   
   
[https://www.geeksforgeeks.org/tree-traversals-inorder-preorder-and-postorder/](https://www.geeksforgeeks.org/tree-traversals-inorder-preorder-and-postorder/)   
   
[https://www.geeksforgeeks.org/longest-common-subsequence-dp-4/](https://www.geeksforgeeks.org/longest-common-subsequence-dp-4/)   
   
[https://www.geeksforgeeks.org/partition-problem-dp-18/](https://www.geeksforgeeks.org/partition-problem-dp-18/)   
   
[https://www.geeksforgeeks.org/largest-sum-contiguous-subarray/](https://www.geeksforgeeks.org/largest-sum-contiguous-subarray/)   
   
[https://www.geeksforgeeks.org/find-the-first-repeated-character-in-a-string/](https://www.geeksforgeeks.org/find-the-first-repeated-character-in-a-string/)   
   
   
   
   
[https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)   
   
[https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c](https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c)   
   
[https://www.simplilearn.com/tutorials/deep-learning-tutorial/deep-learning-interview-questions](https://www.simplilearn.com/tutorials/deep-learning-tutorial/deep-learning-interview-questions)   
   
[https://www.javatpoint.com/deep-learning-interview-questions](https://www.javatpoint.com/deep-learning-interview-questions)   
   
[https://www.springboard.com/blog/machine-learning-interview-questions/](https://www.springboard.com/blog/machine-learning-interview-questions/)   
   
[https://www.ccs.neu.edu/home/vip/teach/MLcourse/6\_SVM\_kernels/lecture\_notes/svm/svm.pdf](https://www.ccs.neu.edu/home/vip/teach/MLcourse/6_SVM_kernels/lecture_notes/svm/svm.pdf)   
   
[https://www.knowledgehut.com/blog/data-science/linear-discriminant-analysis-for-machine-learning](https://www.knowledgehut.com/blog/data-science/linear-discriminant-analysis-for-machine-learning)   
   
[https://towardsdatascience.com/what-is-two-stream-self-attention-in-xlnet-ebfe013a0cf3](https://towardsdatascience.com/what-is-two-stream-self-attention-in-xlnet-ebfe013a0cf3)   
   
[https://www.quora.com/What-are-the-disadvantages-of-a-recurrent-neural-network](https://www.quora.com/What-are-the-disadvantages-of-a-recurrent-neural-network)   
   
[https://datascience.stackexchange.com/questions/27392/so-whats-the-catch-with-lstm](https://datascience.stackexchange.com/questions/27392/so-whats-the-catch-with-lstm)   
   
[https://towardsdatascience.com/the-fall-of-rnn-lstm-2d1594c74ce0](https://towardsdatascience.com/the-fall-of-rnn-lstm-2d1594c74ce0)   
   
[http://jalammar.github.io/illustrated-bert/](http://jalammar.github.io/illustrated-bert/)   
   
[http://jalammar.github.io/illustrated-transformer/](http://jalammar.github.io/illustrated-transformer/)   
   
[https://jonathan-hui.medium.com/nlp-bert-transformer-7f0ac397f524](https://jonathan-hui.medium.com/nlp-bert-transformer-7f0ac397f524)   
   
[https://stattrek.com/hypothesis-test/hypothesis-testing.aspx](https://stattrek.com/hypothesis-test/hypothesis-testing.aspx)   
Z1 \= W1X \+ b1 	\# shape of Z1 (noOfHiddenNeurons,m)   
A1 \= sigmoid(Z1)  \# shape of A1 (noOfHiddenNeurons,m)   
Z2 \= W2A1 \+ b2	\# shape of Z2 is (1,m)   
A2 \= sigmoid(Z2)  \# shape of A2 is (1,m)   
   
A \= 1 / (1 \+ np.exp(-z)) \# Where z is the input matrix   
   
   
The lognormal distribution differs from the normal distribution in several ways. A major difference is in its shape: **the normal distribution is symmetrical, whereas the lognormal distribution is not**. Because the values in a lognormal distribution are positive, they create a right-skewed curve.   
   
![Log-normal distribution - Wikipedia][image2]  
   
   
valence shifters sentiment classification   
We examine three types of valence shifters: **negations, intensifiers, and diminishers**.   
[https://aman.ai/](https://aman.ai/)   
[https://vinija.ai/](https://vinija.ai/)   


[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAfAAAAETCAYAAADJfFTKAAB0E0lEQVR4XuydebxX0/f/f38jUwoZKs1RhEoaJJWQEqlESpRkKJkqogxNhkpISqRIhkZKZSqEBgqJ0ECUIfM8nZ/n9l3ns86+57yH273v7jvr+Xi8H/eefaZ99tlnv9Zae5+z/19gGIZhGEbe8f/8BMMwDMMwSj4m4IZhGIaRh5iAG4ZhGEYeYgJeQvjzzz/9pCJh5cqVwdKlSwv8dna++eab4NNPP/WTDcMwdhpMwEsAJ598crDHHnsEa9eu9VdtN2eddVawyy67hL+99947OPfcc4O///7b3zRj2Pell14KXn755eCvv/7yV+eU7777LjjwwAODTz75JEzr3LlzsOuuu7rrJa/Vq1cPfvnlF7VX8VC2bNkdXh5FTenSpberruQrXPcHH3zgJxv/IWhXMm03rrjiiqBixYp+crGTsYD36tUr2G233YLPPvvMX5WWFStWuAa1e/fu/qqcQz78X79+/fzNio3NmzcH9957byTt8MMPd2Lz3HPPRdKLiilTpoQCPnfuXH91VvTt29fVAzke//fs0cPfrNiYPHly8Morr4TLCGb58uWDn3/+OUwjX3jff/zxh1vGQCpqYV20aJGr15pSpUoV+Xl2NGIEFQb2o67/9NNP/qoSx/jx44MvvvgiXOa633//fbVF4fn9998LPPMlnTmzZwcfffSRn7xdEA1cvHixn1xiwYhLEvCnnnoq0l5feumlwf7776+2yA0pBZwHcM6cOcHuu+8e1KhRw1Xqwgo4+5533nn+qpxDPvbaay8nmvIbNmyYv1mx8eyzz7o8+Hz11Vd+UpExf/78UHBffPFFf3XGbNmyJTj44IOd1ytQwbE8qcC5oEKFCmkNLizn4oYoxi233BJJMwGPQlkUts3INRjyy5cvD5eLUsB/+OGH2Ge+JNOyZcvgiSee8JO3i+HDhwcXX3yxn1xiSSXgrVu3Drp16xYul0gB//rrr4Nq1aq5cCkPcWEfxkwE/Msvv3RhiPbt2weXXHJJsGHDhsh6+ogHDRoUdOjQIbjyyiuDzz//PFz37bffBldddVXQsWPHYMCAAYmFDuTjmGOO8ZMdVDA8PIFzkLZw4UK3vH79ere8bt06d55zzjknmDp1ari9QMVnXZcuXYJ77rknTL/77ruDnj17ujxwnIceesilc06Wf/vtt3BbvIHLLrssOOOMM9xfHSIeMWJEMHbsWCf6V/9z3Z06dQpmzZoVrvfJVMDxlCjbN954w1/laNq0aaw3RRpG3qZNm9wy9eWZZ56JbIOHI+AZ33jjje5ecm3ce4Fzc21cL/WAqM1bb73l1pFepkyZ4LjjjnPl9eOPP7p0/gfKgP958Pgr92bChAn/Hjz41yjlnlBXeABfe+21cB28/fbbLk+sHzduXGSdcNtttwX16tVznj3nob8dEHDq6fXXXx+ceeaZ7q8GQRs8eLC7p0S00nk4CxYscPWod+/ezngSGMPw/PPPu+ujDHlWEAkpkx49egR33XWX21bKkXNeffXVYZnBo48+GrzzzjvBqFGj3HMnEQsNdYZr4vnimoYOHepv4o7BfaLMbr311lDwMYzZ/9prr3V5wxP16zn1YtmyZeHy6tWrXf0GnmvahNNPP92dn2UNzznPIdfGdno95+F6ODdl+PDDD6s9o3A+BJz7zn7S1vGcc70cn7aH/GvwKM8//3x33TzbccYbRgB1XZ55eU7ZHoPhjjvucOUKlOOdd96pdw/LQuBZlrbl8ccfj6zzefXVV10dpxvNjwBwXHlegbKS54h1OGwYqZJGvZGyue6661yefQPWz+vHH38clvvo0aODU045JWjUqJE7jo52+Pjl/Ouvv0aWk6A+iA7wV9cznnme9UceecSVB4LLMyNQx3mueQ4uuugip31JAn7//fcHhx12mGsDpHxEwNE67g/Pgz/WiOvg+ac+owPpnv9MSCngVEh5qItTwKnINH4iMvLD+wfOvc8++0TWsT1QEeg/1uv23XffRK+B9UkCzjoqmEBjQhphY0DIWT7yyCMj59Pi1KBBg8g6fggfVKpUKZLeuHFjl85flqVCLVmyJBKm5seyiA3/E0VAzPQ2r7/++r+Z8MhUwBE6OZdfftQD1iXBAy2Cd9NNNznR0NSqVcsdkx8h72P/KafZs2e7/mrupTwo9913n7suro8HhUaE8yLyCBPphx56qBMuEU7JF40h6dQH/oqA1qlT599M/MOJJ57oIgYYTzRENNyEw4EGlLwgCuStcuXKwTXXXBPuK5Bn1h1xxBHuPFu3bnXp7MuDTT45Pvk4+uijw/0OOuigoH79+q5e84CzfVI/Kw0u10qjSGPN/2K0jhw50jWw1HPE4d1333XPJeXAGIeuXbsGY8aMcdtiWLVt2zZ47LHHXINDIyMijhdBWVIvhwwZUqDhBI5J1AUhwCjgnJxb4BnhOmgwJ02a5LbFwALKhv3btGnj/ufeM05ArpluD+palSpVwuPRuF1wwQXuf+pBq1atXP3lL9cmjTl/yQvP68yZM4OTTjrJlbd0pXDecuXKOSOCRpb7rI1pDQ0+2+N1kk+JHDB2gnWILlEd3W5gPJF3PMoHH3zQrcfZ8eGZ5Zo4HsemjgL1h2eiZs2arh4C4kId0ZBvgfaT41An+HG9p556qtr6f1CH2bdPnz5OmCg76p7APdPhbMpNniOumeuhXSLPQN1jPelnn322a/MoX55lgfLQYMi3aNHC/U/94TmkTDmmNh4E6hIOBPnGWQLOs+eee8bWTQ33jO04Nt2ERFa1DlDHqOvUI55tnlPyIjRs2DA44IADXBuGIbfffvu5Zy5OwDEWqeccU8oHAWcfniXuI0Y35aFFmrJCH55++mkn4Nw/31HNlpQCrilOAZfwPA8alhAPjFQWwENlGSuOfNCPjOULF154oVuHR8I6+iWotEkWHttSQWjY5MfxZF0mAs6PfGJF63wivCzTSNLAIHoixFji7CPHYJ2MPPcFnOOxjAiwzbRp09wylQ7kmJQn63lQWKYBiiNTAUeIKBs8Mp90YcD+/fsH7dq1c/+nEnAab+639laOOuooJwCAgEv5CjxYVHiIC6H7+fJD6CLgNFI85DqKQAPBNZM3/kfUBOoFD7RvzEBSCB0vV5DIBODBYYRqKCMdhtNwTatWrQqXKVsaZUDAOZZu1ETA6aIRGAfA86HB6OD+ANda+//uSxIcU0cSpB2QqBSNkDZgyRPr8V7jQugYI+JlElnB8KOMJA+IINEG7hXCr6GeYJwA18WyhgaZBhI4rw6B33DDDa5xTSIuhI63JFAXSJMy5xmnzRG4Vspi48aNYZoQ9+wg4IiMLvtUAv7999+7//EMBbqypO76UBdpFwS5b7RRsj5JwMEPoYuAE9EQOCbnp32CVAIOtO+ZhNA5z8SJE8NljJy4a9RgoGrDEjiOXCNiq41pcUgkssm9o4wFnAfWxwk4xIXQKVP9TDZv3twZT0BZ+tdBWWBcbw87XMDxrFjn33wsHNKxYBBD/ufX8R+LhwojokvIU/ZH1BF/8czikG3xBOQnrxuxLhMBF48auGlYesANYz0WluBXvLg+cC3gWKZx5UE+SccwEQGXh1kePraJI1MBTwVem59vDQJMKB9SCTjw97333nN5oYwRQ/FuRMA1PCzSMG+PgCO4POR4t/Ijr+zPA43Vj8BTTymzVHU9ScD9MLTkjYaZBkSfG8+LhtyH0Cz3WG+L8IkBh4ATptOIgOv6xv3QXU0gHg1QrtLAJOEfE/DQ8JQwsliv88mP8if8HyfgeMvitXEMohGUAc/xtm3b3PaEPgm/0sDp455wwglheVatWtU1mno94U95Fv18I7a+OGriBJy6qaGRp0EXMcfQ1efHyxXjSJMk4DgamlQCLk4MBqU+J+uli0l48sknC5wPKJ+BAwe6/wsr4H795lnhWQC/zSqsgHMPqRsC+c4E6iP3jPMSlapbt27YjYaA++N0uB5584e2W0Md5HqyEXCpe8LNN9/svHnAO6eN0/dOulO3hx0u4FisrPMrLqFO0nmwgcaVvgkRMh3mpd+Ufg0aJtbxoCUNQGF9piF0GlLSfAHXXhbnkhsn4XMJ/8SRTsDFaMCAiduGxkMEXLxYMXAIOcZBfzTr+cUJOJVbD0yLQxprDQ844STgAaABglQCTj8lliiNOA8z/WI0QqkEnDBhUQg4eSJURsjT/0lZYszxcOH9cVxfKIUkAff7QSVvhBsJPfrnpR/ahxGu1Ct/W7wMQMDpQ9fECTihbF98CR/KdjRC2nuOwz8mEG6n8RHB9fPJD0M0TsBpGPG48VR4XqkP9JvT2JI3QuHAc0DUwj8uP6A8KQd/nfS5+vkujID7bYgIOM836/ES/fP7byZAkoD7YyRSCTjjP3ju/fPx02NIQLrCfDAYRUALK+D+9yroEmCMABSVgBOVkXEBCKP2ahFF34gAIntERXjWOUeTJk1cfrWA+91hrBcBl/NpkvrAIU7A/UFsWsCpO5zDv3f8tocdLuDcHCqTX+FEjHnAqTTz5s1zlYx84IWyDvGngLE4gQaDPifW+Y2rwLpUAs5Nk0aYUB9pmQo4BgbrCScLl19+ufOeZJQ5IX620Q29FnDEWARaw4NMmvQb6mNsj4CLdy4PyYcffligwRYQMx3d4KEmXxhZ/JVr5EH1XxnEIOG49JPWrl07cg76ZnMh4Ahj0r2PAzHn2HED97IVcDx/+t8yAWOKYyWRqYAzAMsfF0FYWEKJmQq4f03cv9tvv939z33Xr/Bp4gQcMOAYbyB9xjzfiDoRNJ5z4BmfPn263i3Cscce68KcSfhlUZQCTpvEegyYTMhUwDHQ8OIFicYB4zTi7kUc4gT4EMGRcQDULylrkGsSkgRcDwwDujHEiBdjQ6CeFkbAMVZoE+hK8QcfJgk4dcr3omknMxVwvztGoixFJeDkTfrLi5LtFvCkQRQaEXB+3GT5Sf8gx2AdgwYoZAqCZWm0JXzEA0+I/LTTTnPLiImEIXigeXeRxollhCIO1iU14uSH9VhK+gMomQq4Fle8LR4C/ue4UhHkQaDC4ZWC3weON8Iyfb/0e8oAPvEGsxVwQvpyLdJ3CTRAPMjcC+6veGcSYvXhIWdbGlceIn4yBkH6Z4HwKfdSDC7CpmzD/wwwwXuSwUhr1qxx15OpgNOHe8ghh7iGRBpof/skAcdI4VwYUYgG5Uc4+Pjjj3frGXPAAEUxZqQBjRMo6gSDgsiH3IdUAo44UHaEhsk35yCsljSwintAo0AZcw7qAOUKmQo40O8r90E8ZukbzVTAqd/cL8rszTffdNchfYcMDOLZk/vBOArKGCNE2gz6XfVIYhp8rodnRGAfniXpzqIcOQ/dRByH54d8yIBDqSf0W7Ke43NM6V/3yyKdgHN+vEnJJ/snCTjg4dHWsD3n4dsEfj3UiIErdStOwIkkcgyunfuuuwyA+sLgKNaxDQZa0rNKG8ozxX3hvpE/rlEGXGJQ08XCccgTYyz0uWj/ECjJr7RblAHXzH7SfSmDErlGeWOJcmJZCzjtOGMuJE9JUA7kn2iovofUY79LSCBihpEsZYPTwvkzFXC2lW4f8ke+WZ8k4LTFGFti0KQTcLSK44mDRD4Zs0E93h62W8CpQKluBmgB1z8RcJARlvLjhujjUtlFuPjpfhEsd70vXm8SrE8ScLwCbgrb0BgTIuP/TAUc6I/i4ZS80GjosBrlKAYKQg2+gAMVRF8TRoWIQzYCTrnqEfzsS2WW6+RHGBvIO+t5eJNA5KQbgx/HwbLXrxeRL+kC4cFgFC7CKMIlDwciT+PPqO5MBZwGnvJmGz34UJMk4EADI+XPj4ZYPGweRjHcqNfkb8aMGeG+GqI/GFhsK908qQQcqE8clzIhPVU5kyf5wA8/3X+XjYAT/RGDmXsrnjNkKuB4QVLnyL/vFcn9lHuqPTu8KblegcaMNMZBCFwf/doaIlmcV370i+rrk7cI5Pgy0BH8skgn4BgGEgmU7qJUAs79adasmduO81OnUn2IiYgU20q4OU7AAQ+V7WgbiUDoNhLjD6NfyhrBIi0ODBueLdkW8devdnIdRLNYJ/VC3yPqtDznIAL+wAMPhOVNeUxUr2gSEZH7gZHOa1RawHEApG30X7HSsB31yB+IzLPLMxcHbSflIfUc4wQDO1MBf+GFF8Lr5Ri0Z9zTJAEnb9QnKZ90Ag44lVyXlB/Ppt9eZEvGAm4YSdBQ+sJhGMbOQ1IfeHFAF40fqgeMlw0bNvjJ/2lMwA3DMIyU5ELA8UyJriUNqGXd9nqsOxsm4IZhGEZKCMnThVGcAkq4W48419B1hsDLgGXjX0zADcMwjBINYw545dK66qKYgBuGYRhGHmICbhiGYRh5iAm4YRiGYeQhJuCGYRiGkYeYgBuGYRhGHmICbhiGYRh5yE4l4Hz3WT6LyaxGzHKUC3i1gU8HJn3WMF/h+8l8AjLVJyiBbyH7k2ZkCnOqJ32u0DAMw0hmpxJwvvktAs43h2V6y3Qw606qOcTTwccN+EoR38cuDnj/sSg/Ich3fPmmeTq4LmbpYaKDJJjhSyaQKAycA2MrboYhwzAMI5mdVsCzgQ/dM5FJYSluAce7f/zxx/3kQoO3rCcuSAWTlTBjWBxMOkCZby98npGJFQzDMIzMyYmAI2zMbsMUhLfddlswbNiwiGAyReHKlSvdNkOGDAnn2cUrHjdunJt1CC/Z/w4vsx4NHDjQTcn4448/RgSc2W5kakJAbJhJhxlpZJ5hQt9MtUmImPPoWZT4dCAzNQ0aNMhNs+l/AYiZya699tpgwoQJ7pzpBJyZdJidhjlxdciY6eX88DMz48gsOeSJWca4TvIqc6NTRsztLdfvz9Ptz/ZDOPzdd991s1aNGjXK5ZfjMQNZEtwXtmMaPz5xyGxG+lOKTO2ZNL2f71HHTcmpYVrPadOm+cmGYRhGAjkRcKaIZM5lpiJkqk+muGTKNxHFc889100nyfR0zC9NXzKiiPfHtKFMdceUlUwLKQLC3LpMkceUfF27dg0qVqwYlC5dOhRwpvoj9AwyBSVTRyLUTOPJ3LQSvmWKt+OOOy6cVpNwNXlh3nH60elb5yd8+eWXbuq5wYMHuzmN+Z/rSRJw8se1MA8y00ByLBFxDBqmddQwxzJGC5AnphRk7mHyytShnIv/KQ+ma2zatKkrC8oE5s+fH85xLbAN06AyN3PDhg2dMHOMMWPGRLbTjBgxwk1pyXGZFpDyYEpMWLdunTuG/+1i8sPk9czZLfNQy/lSwdSNMj+6YRiGkZ6cCTgNuJ7flZAwc98CAs56PYUcYWPm+RYQWwwAPEmoVq1aKFggc3fHCTjCw7HEYOBYfBhflv0Qes2aNYMePXqEy3i9bC+eO/NNa4938eLFiR44+yK4W7ZsCdOYG/uCCy5w/6cTcPBD6BwPI0SMGa4DoceLhVQCDpmG0JlfF4NHDJvHHnvMDToDvGWO4UcmgHxhUHENN954o4tmZDJQjTL2oyyGYRhGPDkT8NNPP91PDkqVKuX+IuCNGzeOrEOkEAj/R9ga4gQILzdOwNn2o48+0ptG0AKO1+ifU349e/Z020i+hVR94P369XMD6jSEwCX/hRVwf1ael19+2R2TvBSFgMs13XDDDWHaqaeeGhou5I/yTmL16tVuf99DTwXbv/fee36yYRiGEUPOBLxRo0Z+sgsNQ5KA47Xiuemf9K36AkQ6wpok4PSvJ6EFXLxt8Rr1T8RI8i2QniTgt9xyS9CyZctIGn3xqQQczzedgNM/r5k7d25KAaf7IBsBZ5Q63Qg//PCDW6ZcKF/xuO+//353jDgPHDAwKCe6OHzoFlm2bJmf7I6noyqGYRhGMjkTcERRD2TCQ5PXk+IEnH5p7f0B/c1bt251/zOwS4fcFyxY4AQgTsDr16/vBs5pKleuHP5ftWpVNyBMIPxLv7uGPmARaM6tB3ORniTgCKDum4cTTzwxOPnkk93/kyZNcgaEhv56X8AJZQsIuOwvEJLnmoFuBn9Ud/ny5bMScPrUtWGB5633obxYjgt5Y9BguNDnL6PUdXlJ37gPBoNhGIaRGTkTcEQPIXruueecsCLQ0qjHCTghb0SfUduMuGbwG2Igos16BnGtWrUqePjhh523hwDECfiKFSvcsRAwhAixGz16dHiuxo0aufxNmTLFLYsxgAjRb9+lS5eIuODN4yUzghtvGs+UX5yAQ+3atZ1I01eOp4wAMyoeGDBH3hihjnFCpIJjaQFnIBn9/3jdiB/7k99u3bq5/DESnWNs2rTJbY8XyzLH4HoJfZN/EXAGwrG+U6dOwaxZs8LzaHSfPyDYhMy14YChoQ0TDAPKo02bNi4igkhTjtwfCfnzzrhvfAH964T5herVq7uuBMMwDCOenAl4586dgzlz5rgBUYiAiCUwIvyOO+5Qe/wLQnnmmWe6Uc2IroieMGDAACfUHTp0cK+oIWQiDIyuRrgFhP6ss85yInHJJZeE6cA+GAgYEgIhdY6L5927d2/3mpqG0PAJJ5zgRJBR7nib8vqbD4YKEYBWrVq569eD+QCPmbAyEQleTxs6dKgrKwEh5PgItoxCxxDgegmVk09GcWveeOMNV3aIN+H3u+66K5g5c2a4nnNyPMre5+uvvw769u0b8ZrpUuBadRoGgjY0OCbbcC8EIhkjR44Ml7kPDRo0CJcFIiLr168PlxctWuSMAcMwDCOenAm4jJA2th8EnHfTSwJEQdK9463hNbMHH3wwkjZ16tSIt28YhmGkxwQ8DylJAk4fOIP0/A+3JEHEhA/KCHjtfKTHMAzDyI6cCDj9n/379/eTjULCh1HoTy9J8IW3dEifOHmXwW+Z7GcYhmEUJCcCbhiAgNNv749lMAzDMLLHBNwwDMMw8hATcMMwDMPIQ0zADcMwDCMPMQE3DMMwjDzEBNwwDMMw8hATcMMwDMPIQ0zADcMwDCMPMQE3DMMwjDzEBNwwDMMw8hATcMMwDMPIQ0zADcMwDCMPMQE3DMMwjDzEBNwwDMMw8hATcMMwDMPIQ0zADcMwDCMPMQH3+PXXX/2knLNixQo3d3amvP/++36SYRiGsZOTVsD/+OOPYMmSJcHw4cODt99+O/jtt9/8TdLywQcfJP6+/vprf/Ni48MPP4yce+PGjZHrYXmXXXYJjjrqKLVXPDNmzHDbPvnkk/6qWLZs2RI59/r162Ov/eeffw722msvd2wYM2aM+3/o0KHelv/Sp08ft37SpEn+KsMwDGMnJqWA//7770H9+vWDGjVqBAMGDAiaNGkSVK5cOfjll1/8TVOCwCT9br31Vn/zYqNs2bIFzr/ffvsFGzZscOsR1L333jvo27dvdMcYshXws88+u8C5+d15552R7bp37+7S77//frecTsChVKlSTvSzvS+GYRhG/pJSwN96662gfPnyzgsHwrp16tQJ+vfv722ZmtmzZ4c/xOjggw8Ol9etW+dvXmwg4LvttlvwyiuvuB9RBfKDaP/5559uG7nWdBRWwEeMGOHO/fDDD4cijkcOhO933333YNdddw1D6JkI+Jw5c9w2ixYt8lcZhmEYOykpBZyw72uvvRZJa9euXdC6detIWjYgNNWrV4+kNWrUyHn6zzzzjBOwefPmuXTSLrzwwnC75cuXu7RRo0aFadOmTQv22GMPt9+hhx4afP/99+E6HwQcb1XDPuTpp59+Cr755ht3/MGDB4frV65cGRxwwAHu+BUrVgy++uorl+4LOGXFvvy+/fbbcH9BBByjRRDh5ViwevXqUOQFEfB777036NWrl8tHhw4dwvXwww8/uG3KlSsXpnHsffbZx4XqDcMwjJ2PlAIeBwI1efJkPzlj4gR8zz33dOl4nlWqVAnmz58fbtu8efNwuxdffNGlXXXVVW5ZRJHjnXjiiWHf8ebNm8N9NL6A//XXX07k2Ie+8C+//NL9f/7557v1DCZjuUKFCkGbNm2C/fff3y0TqtYCLqF3vHtEOI44Ae/Xr59LW7ZsmVuW/uxNmzaF24iAV61a1RkqRERYvvrqq8NtgKgG6RJJOOaYY9xyrVq1ItsZhmEYOwdZCTghdRGwwiKCqxEBf/PNNyPppKUScF+gGI1NWsuWLcM0jfSBsw8/Ee8hQ4a49b6AT58+3S3PnDnTLeNlr1q1yo0NEAF/6KGHgjJlyjjxZpBfEiLghxxyiDu3CHHNmjXDke8iuuLlgwg4BgID3IDlI444ItwGGJ9AOnkDBuzdcsstsdEAwzAMI//JWMDpk0XwCM1uD4hMkoD7kJZOwBFOPGP5iUDHETeIjevBEwdfwL/44gvnsZO27777BnXr1nXCCCLgRA34e+SRR4bniSNuENvJJ58cMYbIN+mE8gUR8K5du4ZpnJNwvuakk05y2xXmLQHDMAwj/8hIwBEFRFaLSGFBZAor4ITWSdMCfuyxx7o+a/0bO3ZsuI9Gh9B5VYz99eAwX8ABw2XYsGFBpUqVQjF/5513QgHnJ6H7F154IdzPR4fQ6bfnfwwizamnnurSP/300zAtbhBbnIAfdthhbjvxwA3DMIydm5QCTt9u7dq1g06dOiV6dnfffbeflJJsBZzf+PHjgwkTJoTbiYCLR41gM9iuWbNmbpnQdxx+H3jDhg3d9jIQzRdwRtuzPHDgQDdSnPOyfMUVV0T6wGV0PsuPPfZYeHyN3wf+yCOPuOXjjz8+3GbNmjXh+YRMBPy7775z23CvhB49eri0a6+9NkwzDMMwdh5SCjgDpRABwse8L61/Auuz8frYPlMBF/GS37nnnuv+ioDzOhYhdL0NYe4kfAGnT5l9GNkdN4iNEe0HHnhg5PgMJmNbfxT62rVr3bJ+BUzjCzgcdNBBLu3VV191ywxAk8FwqV4j8wUcA4dtGDEvtGjRwqW1bds2TDMMwzB2HlIKOH3DiErcT8D7ywZ//6Q04aOPPnIfO+FVKUSN7aTPGvj/qaeeCu655x43yCwVcefx0/zjc86PP/7Yvbr22WefhcIqedFiLceKE3ApS70u7nouuugiJ7zyKlncNn6exQjR77Cz34b/+0CNYRiGsfORUsCN3IMIE+HAO8+Em2++2Qk4RoxhGIbx38EEvARC1EF73OmwyUwMwzD+e5iAG4ZhGEYeYgJuGIZhGHmICbhhGIZh5CEm4IZhGIaRh5iAG4ZhGEYeYgJuGIZhGHmICbhhGIZh5CF5K+DbM6WpkRnvvfdeZGY0wzAMo+SQcwG/8MILgz322MN9+pNf/fr1g3HjxvmbORBp+e45ML0n82DzrXBmBNPwHfNJkyZF0ooD5tjme+pJv6RJX0oKzJhGPtu1axdceeWV/upg06ZN7tv3fNmNr8Lxmzx5crBx40a3nilQDz/88Mj0qXwLv0+fPuHy9nDjjTe6b+PzSwWf10336VyBuuEbfHy7/uWXX46kce2UCd+Zp4ziPolrGIZRUsi5gEP79u3d5z+BGc/KlCnjRF2D59eqVatIGjDbl+zr89VXX7ljFyd8upTvvzdq1MhflRcg4JRfnDhxXWeddVbsOkBcmZUNmFiGyWC4Tw0aNPC23D7KlSuXcha1Sy65JOuvz51zzjkFBP/tt98Opk6dGkkTksrIMAyjpJBzAadR9GcfY0pNlpk0RMAL9L0mOOWUU4Jq1ar5ySFMbzp69Gg/ucjYmQX89ddfj8xoplm6dGk4fzkeN5OuwP777++EvKjYtm2bm20t6ZgjR44MevXq5SenhevFUPSvm/qk518XksrIMAyjpJBzAWdWMRrHm266yS0zqxYiQDhdaNy4sZtz24dtCW0yKxjzf9PQM0WoD+nffvutn1wk7MwCznV16NDBTw6nXWXa0xNOOCGc5YwIyeWXXx7d+B+uueaaoHz58q7LA7Zu3erucSb07t3bTalK/g477DB3LzHmgOldWfZ54403guOOO86dVyCkv9dee6mtguDRRx8Np4oV8ML19K1CUhkZhmGUFHIu4DK/9QMPPBBMmTIlqFSpUtClS5fI9JiI8rvvvqv2+peXXnrJ7Ys3OHbs2ODWW2+NDacjHg8++KCfHOLPIR73S5rjXAT8kEMOCW644YbIb8GCBf7mJY5UAg5EQVq3bh3MmjUrePPNN919qFKlijOwzjvvvMh+iPLq1avV3v+jZs2aQefOnd3/v/76a4E54JNAtOnfxoijTmDoyT0mhB93v4EJYDDuli9f7urV3Llz3U+DISLGgYbuGz/ak6qMDMMwSgI5F3C8KRpHGn68qZNOOimynsaedDx1n6uvvjo49thjnXDD2WefXcDLAgbGsW0SDGpK90tCBJxz4FnqHx5iSSedgAs//fRT8OGHH7p50Nn+mWeeiazHwOE+EfKOg750xFJo27atWhsP4yE4V7du3cLBgAy2k+MQHcD4SuK0005zYfK77rrLX+VApDm+P7KeroHnn38+kpZJGRmGYexIci7gNIx169Z1/9PgIgISkgUEgW18jwgY3FShQoVwqk36wocPH+5tFQTNmzd3g6yKg2xD6PPmzfOTsuKtt97KampRgX1kFLmObmQq4MA2Bx98sCtzH7ooOA5lEYf2lj///PNg5syZ3hYFoVuEffD8BSI0t912m/v/1FNPTTk6nUGM7M81x4FxyHq/zxuj8p577omkZVpGhmEYO4qcCjheKg2j9JvS0LLMCGENjfQnn3wSSUOE2PbZZ58N01im0fbhVTPpY4+DV6Dq1KmT8pckApkKOI1/s2bNikQECGlnexy25/U8PFbyO2DAAOfJZiPg1113ndsfbzwOoh/r16/3kx1PPvlkeB4GnWkjIgkGxu23335h3hB+jiHjGRh9LkZBHPSfk9+kQYzigfvXQ7nQF67JtIwMwzB2FDkV8J49e7qGUb9/y3vFpOlXfKpWreoGHGnWrFlToK8S7x2hnT9/vtry35Do7NmzI2ma6dOnp/0leb2ZCjjvuyeFlwvDUUcd5Selhe4GQspC6dKlYwWc/mNep6IPX8TykUcecfuyTli8eHHEsKlVq1bia1iE3zkPXvUrr7wSpnOfV61apbb8H4gvbxEITz7xhLvHiP/ChQvdfU4ScLz2JUuWOPGuXbu2SyO/Gow9xi5oCNVzXn/Mg19GhmEYJY2cCTgDiBBWGkYdMsfzIe2AAw4IvbTx48c7Edc0bdo00l9Ow8ugJQwAv/FFqJI86O0FAR8xYoTLH2Lh/7gGBB7xFBACQrTSDzx06NDYCMHDDz8cnHnmmWFfOn2+ch2MsMaIyQa8WTxvoH8ZoYoTcLoiyDOj+3v06OG6OChDGQtAnzGjt33hZSCif58Ejs/5zjjjjEg65+L8ceMMSJcPxgADAxk8xz2WcsCo0MYV3j3jEfRbCxwHA8sPlVMWfpcLRov+KI3gl5FhGEZJIycCTt8jnh0iwo//9RfLdDrQcOJt6waU9X7oc9GiRQVCs4hg3CtoRQWGiOQ37gcI3aWXXhrZj3zyvjtfcmOkd1wfP9DPy2tRgGhLF8GKFSvCj5vQFTFo0KAC5cHX0wQEGRFiJDYfv8F4Yp84Aec4lCX78zUyrpE8Ykjx3jVfvYsziDgGQp8UrdDhcAFjgEFmXIOGY0j5CZQRXSb6HhOZoYz0NnjaOg/r1q0rcCyga8avL4zDIFrg45eRYRhGSSMnAl4YEBFGIGcDQnTooYf6yTkH0bz++uv9ZCcy/hfnfOh3llHU2kvfvHmzG3UPvIuNsDHADI+S637ooYci4wYmTpwYK0JxAr49IKC8gy3Hw+DgOhHGuFH5COjpp59eQEizgbEF2XZPYNT5/fV4+/JGg09RlpFhGEZxUGIFHGhgCTdnAh4jo5VLQqPLu8j9+vXzk4MNGzYEBx54oJ8cgQgCr2DxqVA9LoCJRS6++GL3v75GhJsPqvhjBtq0aePeqfbhYywYOXj3Mrp7e8GbHjJkSBg2571xvOA4+NJbUvQhUzgPXRmE/DOBqAI/Dd8UoAtAQ3lTJozVoIxKQl0yDMNIokQLOGTa2Pv94DsSBO3EE08MlxECfrxTzfvUeKYiPhgoP/74Y7gtoXIm00AQNQh70oCxOBhTwD65gm4SIFLgh/aLi0y9+Li6ocvcMAwjHynxAp6vHH300aFwDBs2zA1qQ8Tp++/YsaNLR4Dol73jjjvC/RgHwBfOfO+PV+P8tDjw1Bk4RgiYwV333Xefv4lhGIaxE2ACXkzgheNxp4PZvQiX02/MAC9GXvsfR+EVKfquDcMwDEMwAS9G6FN9/PHH/eQIMgMbIs47yvo1KuDVsbjvwhuGYRj/bUzADcMwDCMPMQE3DMMwjDzEBNwwDMMw8pAdJuDMVpXJqGpN+/btwz7jkg5zWjMZx/bCe958V90I3Axz/gC/fIK3EqjDfKRoR8CkQTIYkm/UF+VrhpMnT3bjOGDGjBnui4NFBe0E7UXcFMPZwmub/tSxhpGv7BABnzNnjvuOeTbIzGU76v3de++9N6hcuXL4o0FJekedL6KRVz6asr3w1bXbb7/dT/5PQpnuKPErCnjLIOnb8blAf2ueLwLyBb2iQKZxffDBB90ycxZMmzbN26rwyCyGfAhpe+Fzwsw+l63zYBglkZwLOB/5YBrKuMksUsHrVoWZkauoYM5oPEA+vcmP97z56pj+prumMA0EXwaTz6UCxyhbtmzWZZUEx/O/0Z5PMBFKPtO/f3/3lbdMINLAxC2FqUdx8H0Bnjsh0+NS7/kWfjrkeLwOyQxyeha7bGFWuXLlykXSMs1vOmQe+7hv+xtGvpFzAa9Ro0ahPi7Cd9H79Onj/ufVKpn4xAehT2Wp4wXpT5TyTW0aqHQNBI2S/v42x6EhSMqHD+f0Z8fyQaC058K1cA6MBM4jIUofohJvvfVW2kaJObqXLVvmJzt4VS3d/hquZ+3atWnLrTBwvf6nWAmfMrFKOrj3SVEajpnuHmQCIsW9yQTqidQRvF5/mlu8S/0Ne4Fv2TMznQ/ebtzkK3FwPyWfV199dYGpVDWE9/kIkH8/qfd+GnDsOGEXA52PFFGnsv1mPTCTXOvWrf3kCJyHSX/ivrcvrF69OvZrfRhGmX6G1zBKMjkVcOaFZvYqaRD4CpmeG7xSpUrBKaecEi6//vrrztNle5krmvAXouaH0+nP0+sqVKgQft7zxRdfDOrUqeOMB9bRD4ZIMBmIbM+0lbNmzQqPp6ER9CchYWIM9osLoxMpeOKJJ9z/iBwNhpznoIMO8rb+d6Y1zs96uhYoF+AYlAfHk/3xiAQaXcL5+hokjOnDtVOGbMM0pcDsaBhF3BP2p7FjEhY9FSpUr149/A47/fp4R3JOGmuuMRO4PxpE5corr3T/c4+YxpSvxyEaHJvzCtwDMT64bsqROcuBmdYQd7kO9tfv3yOSlKnkmTA210q9ojy0gLKOyVKEMWPGuHrC5DFMO3rMMce4cuQ4zLYWJxBA3eSTuJIfvsbHX4nY4GXqekHepb5yXM4h9ytue/5PJY4YvJLPmjVruvKSiVsYUyGT/lA2CLscl3uE0UFepE7yV74WyHmZ3U7WUYadO3d2XUzAjHbMFse9lGu/7LLL/s1U8O+kMv6YDr6dP2HCBNc9wnPGfuRd7i9pfENfoH7KtfHTUQ2mjCXKJMfhr2/Qly9fPtLuGEa+klMB50HVs3TxEMqAEgSRxqNx48bheib+QPRpJHgY9axXNCjyEDJ9pzQ8QENAA7Nw4UK3zDScnIu5x2V/tteTeWD1x4kr0OD4k5DgHXXp0iWSJpBXrkcMD74PDiyLOPsgShIylTwy4AlBF4GhwaZBBDwg8jR48ODI9pw7DrbRxhOwLY03jbWkI17ScArsJ40g5frcc8+5/9mHxhDjIB14rUy6ouHaGA8BnJdjyyBFyoP8ybVjQCCKeLOU6fz588PjsB0znMk1MHd58+bN3f9cG+cRQwsPlv1nzpzpllkn58SIoA7IJDAcj2XKnclf2FY8eDw4zhvXJ49II8IYJxyD6AHnFAOGesqyTLDCNhxbJu6R+k6Z8z/3mvXiTZOW6lv3CBqGnRgpGB4cT4yHWrVqucGRgHHInOhCgwYNgrFjx7r/uUaeHSlXyp/jYMToekqdFIFlf66NrwcC5cw+YmxjSGOYa3gmZDpYrlV7/VI28j8GLG2ErCfyxPG/+OILt8y28hyzDV1QU6ZMccsCzxllaxj5Ts4EXBpeefCAZWk8eNAQPRobYJQs3jdg8fuD3hBn8fx4gPHOOIf8ateuHUyfPt2tx4OSBgsIQfbo0SOyPaNo/X43gQYBjx4Dgh//00jGeV8IhnjJHJfGiJmv9HXHwSdU+d65hn2Z/ESgkROPDE8GI0JfA15nkoAjGiL+gEBS/tqLo/HEY9FzdSNYckxGMDNHuD5n7969MxqYNWnSpAIhV/IjE59wXn2PpCGX9eSVsDJpemYxRI26o+cDp6/58ssvd/8TZUB8NU2aNHH5Bu6VDOzCIMJ4QXyBrh4JY1MHtFePcGtR1PAJXd+4wxsnuiD49YHrl/qKkZFk6AkNGzZ0z4gP+cHg0pEhxlbIPZR7zHoZGMr6uO4T0sX4BKIIUv80pEk5cH98weR81BWeDc6nx3QQWSBNyuP++++PPOtElOR5wpOPM7LZH+McOL/u3iCqt3Tp0nBZ7lvc9RpGvpEzAdeNiMCDSsiNUdZ4uXqULo2QNNSE4/AuNXhzCCjCy3HjftIw879453DVVVcV2JYfoh8HDRQijMfPz++f1dDI4aEIci6uhwZc50PDBCb+aHMacW0kME0pxxKvxM8/Pz9MLXBsLQrMOe4bLJQXQqnPyWh7vD3Ay/XPx69Tp07h9kngYWuRZdY1EUoadm1cAIOg8NYEORf50yCwLVu2jKTRaEtkh+P6r55xXBFjhBVvjBAu3QlEfDgPeeWeyauApCFCAhEA3yAFlulWoD9ZQxmKsYrRRMSH6AX3AMONfIroYpyxTuB+8DaCbE+EgHPHvaZ43nnnhd0dAl0VYjxQj+X6gOeH+s35Catro4BJd7QhgHftT++7ePHi0NBmW8Ra32fKjPNhiJFfjF/NJZdcEpm5j3spY12A8D/GNmBgxT17HB+jkzpFd4pAubFOlxMGAvXDMHYGcibgCDSesIZQFu+MYmHz8NPQ0tDh+ej+T8TN/x44Dya0aNHCPdgS0tM/IHTne4h40Vji/vZ+Ywz0fcq5MgFvEBHXYO0z2IbGhW6EOGhEdWNJmFobAoAI09iJ54QQ+PmPuwbgmgcOHBguE2K96KKL1Bb/hjuZR1yDpyeGBfePxtk/X9I5NVqQgPsmojJu3LgCZYwQSd8moW0MFsrQF2REVsYbCNJoUz565DUQUdCePQKBscDxuTYJyT799NNBt27dwv38EfCEx7t37x5JA44h90bAiyWNciLyQZ5kXnSMF86lr59ylv5qQBTnzp0bbk8Eye8OERB2+u013GsZ34FBRv+4hjzzPHC9lIMcl3NoMLi0EQOEs6UekWeeaQ2RFxFour3o3tBgTEmXDCDweqAhzwyGpXjOMk5AwIDgHgIGpn5ThX38ekd+qU+GsTOQMwEnpIgYaBBpPGlprBBbvEQEWR5UGm1p/AT+lzAbD63fP80gFnlIaWh9750GRYsAoeGzzjpLbfE/EBHCsJmCEBMCRQiw9HW+EVGm+vTBW/MFjIE62hMB+nUZSS4Cric+oc9R9wP7EHrUo9hpnP2P4hBWlkFlQMhbixHGhw4ZY2hJHmks/YlYBHm1SMBb4v5JPzSGygUXXBCuB4QIYQOuGw8UELeTTz7Z/c+1+h45AimNNnWI9QwSBLbHQJHwOdDgUx9k/vUN/3jjXDP76RHOfv3Aa457z18EnOMIiIZc/+jRoyNdJeSRcQjasEPgtdeoB1DiVRKRSop6sC0DuQTmkCc/Irx4/jJ+gOdRh5sxpiVvGDi6TmLksOzPrU59l3LAUNBRI55n6hnPAjRt2jS4+eabw30losTzAoTWWRbjinoneZAuFW3Iy1gCGdmPoa+NOQwGmboXOA/5TXpLwTDyjZwJOA2XL1L0RfLQiecpVjYhPwHvDK9WQ9hPvHleZ2IfGnnEB6+OcK00IjRovoeGMCCuffv2dQ0hnoUvZgJhdfrpMoHGRa6RxoWGmGtkIBB5kv59H16LYz8aOOmvQ+CkXw+4Ht3/y/EoA4wVxI8y4jhJYOTg3TAwCaH17wXgLdHAEdZEsMiPbrR5C+Dwww933QJt27Z1oiNfx8JIktHtPggnoV+MIQw5ttWCTh3Qr/YB6+Ue4oVJuRD6lq4ABMe/juuuu84ZMgLeGddNBIhIDNeluwgQV0RGwr4iGtq7plzF2IC4vlwNoWaMCPpsGZGNkUBEAWScAuXAeuor946uJIF6Q9nKIEvKgq+oca8RSeqr388sIGbUHeo1HjUGI8tcsxh+8goVhgADEBlYitBxj+Q5EAE/7bTT3DLjD+rVqxeeB3hudfnTfcNxMAIw7DivjE4H7g1lzfgE8oXxzj5S9gxE43gYdCNGjAiGDx8e6eYhL5QF32Og64S6KtEGGdOhIRqgy4koSrrZAQ0jn8iZgPNOJg+n9hDxUrRwsg6vT3t5hB/1oCVAgPS7sDRIPPBY3PSR6XPQ+MYNWOHBxlCgX9IPy2mS9o8DC18P1ELE8ZhpiNK9a855ECS5dl+MaeQ4hjR2bEdjxDVgoKTzKvB6GQwI5MsfUAYcm1Atn8EkKkAj7r8iRl8xDTFGjZQLDS+iEzciW2AbPEO8Mcpbzi/3XJcN16av3y8LtufchFr9d+NZ1t4r21E+1A0GSfqQpreX/Oh+XNbrOoAY+nnScH2MEOd6CZVzDh3xIVrCQDsGrbGt/z49I+9ZJyOrMQx52wBxpmzIX6qyZkAn5+YYGFjyrEgdknNxTXQRcT8Z2+EfE+9cRozzjOnBjUCdi7uP1HmMAv9da85PnaVLhueU+uXfE85JPSX6wXOvX/Fjf/rw6QqifMVzB/Lu12nukTzbRF0k1G4YOws5E3DAIsajNnYu8DhptA2jJEL0D89dG2WGsTOQUwHH67WHyDCMXEKfv993bxg7AzkVcMMwDMMwigYTcMMwDMPIQ0zADcMwDCMPMQE3DMMwjDzEBNwwDMMw8hATcMMwDMPIQ0zADcMwDCMPMQE3DMMwjDzEBNwwDMMw8hATcMMwDMPIQ0zADcMwDCMPMQE3DMMwjDzEBNwwDMMw8hATcMMwDMPIQ0zADcMwDCMPMQH/j/Hrr78GixcvDpYuXeqvMgzDMPIIE/D/GJ07dw522WUX99u4caO/2jAMw8gTMhLwv//+O1i9enXw559/+qtKLPfdd1+w5557BnPnzvVXhfz+++/BAQccEPTp08dfVSgWLVoUHHTQQUG/fv38VRnx119/BfXq1QuOOOII938qvvvuO3cuftmwbt06d+wvv/zS/Z9E48aNgxYtWjiPff369e48I0aM8DdzLFiwwJX1hg0b/FWGYRhGMZFWwG+88cZg9913dx4bfy+99FJ/k7SsXLnS7ev/9t57b3/TIqNbt24uzxdeeKG/KuSbb75x21SpUsVfVSiefvppd7wePXr4qxzkRV//XnvtFVSvXj34/PPP3XrEctddd3XHSGcsffvtt6EnnQnnnntu5NwIbpkyZYLDDz88+OKLLyLbzpkzxx23efPmbvmDDz5wyzfccENkO+Hjjz926zGGMPYMwzCM4ielgH/66aeuYUZg8NoQPBr/ZcuW+ZumZMWKFaHY6B/HKi7I79q1a9MKyubNm4M//vjDTy4U6QT8ggsuKFAG/EqVKhX89ttvbpsffvjB/dKRrYCfffbZbluMJgyH3XbbLdyfZV1OGDSkSz7SCTicccYZbhvyZRiGYRQ/KQX83XffDcaPHx9Ja9q0aXDTTTdF0tIhAn7eeef5qxxbtmxxXjqC8dZbbwXjxo0LVq1a5da9//77wYQJE4J58+a5kLcg269Zs8atRzx/+eWXcD3GB9tgdADhYpYR9pkzZwY//vijEy3SPvroo3A/ePvtt10IfsqUKS587EN3wpgxY9xgMBFeyFTANZdddplLmzVrllt+8803XZ4EPPHly5cHd955Z/Dss8+Gohon4JQd++IR+4iAa8i7HEPKjn5xlkuXLh1uJwJONObVV191eaHcNZQF29SqVStM27Ztm7uun3/+WW1pGIZhFAUpBdyHflPCrp988om/KiXpBPzWW29169u1axcKCr9OnTpFlk844YRwH5abNGkSWV+uXLlQUIcMGeLSRBhPPvlkt0xYmL+fffZZKGAM7AIE/ZRTTokck98VV1wRntdfTyia0DcURsCHDx8eyWfZsmXdMoYGBsuhhx4aOR/eMvgCPmzYMPd/hQoVIkaFECfg5FuOIfsMHjzYLT/wwAPhdiLgUnbymzZtWrgN7LvvvpFzkBeWL774YrWVYRiGURRkJOB4rzTGe+yxhxuwlC06hE4fr/zwqkEEnPDu1q1bg3vvvTfcnv5YvGnpG/7pp5/cPrIeT5AQf5s2bdwyEQJIEnAMEM6BZ+sLOB4sywzY4pwIF9dMGuKOp04e27dv7/afPHmyW4c4QqYCfvPNN7sf28n4AvFStYDTVcH/ffv2dWF+PP8OHTq4vGgBf+aZZ9xf7pGOUmhEwCtVqhQccsghQcWKFZ2XTVrPnj3D7Tp27OjSMHAEEXB+lJ2UE8aLhmOTLowaNcotv/TSS2orwzAMoyjISMAJNyO2LVu2dEL61Vdf+ZukRAScflcEUX7z589360XAtYfOtuXLlw+Xa9eu7baRAV/8TyhXIAQsIgNJAv7888+H+/gCfuqpp7pl/XoVQq1HhCOeiCShbjknwgmZCrj+cZ2EnwUt4FyrGDuUDeF8yYsWcPnpLgQfEXD628UYwph54YUXIttJOelBdCLgGC6CGB56/ADhcz8f6cYgGIZhGIUjIwHXdO/e3YlMNmQaQh85cmSYhrARPhZ4vYpt8ACB/3ltS7Pffvu5dEgScAl3gy/gNWrUcMtff/11uI2GV7eqVavmtjnwwANDIRRDI1MBxyPt1auX+//888+PbKMFHJYsWeIMJwlPy/XFCfjtt9+uDxVBh9DpRyfPLPfu3TuyXcd/PHzSxVACEfDrr78+TBPvXQt45cqVw3MYhmEYxUtKAb/rrrvcgDLN6NGjs26ki0vACS0LmzZtCoUMCiPgF110kVt+4403wm0uv/xy9/4zgjpx4sRIPl9//XW3nK2ACxKeRyAFLeD8XnvttdCjZQAZ67hWLeBERAhn8z8Dx+Lw+8D1/nRTCNIHPnXq1DAtbhR6nIBrA0rgTQDDMAyj6Ekp4PR1EiqVkc801vSfaiHu2rVr+H8SIuCIDO8Ky+/ggw926wsr4PwQZvYjHMzywIED3frCCDij4VkmzMzANelX50comGPxf7NmzYLvv/8+9MYLK+BidBCCl1CzFnA8dTn+U0895QwWlhkH4A9ik3EDMsjNxxdwkPe9KWvpO2dMAWn6Hf1MBFzu8f777x9uc9hhh7k0uSeGYRhG0ZFSwIEQr4SK+THyW/drsi5p4JSQ7j3wwgp4//79ndjK8Y488sgw9FwYAQdeldLHRKgQWuA6JUzMdTP4i/8LK+BwzDHHuDQZ0OeH0E877bRI+WP4gC/gIAbFPffcE6YJcQIOfEiGdBn8BzJ6XO5rJgJO1wrLOppQv359l3bLLbeEaYZhGEbRkFbAgQFNhIvjXh+Le2UpFyAMTz75pPufd8b5IEtRgXDxPjXvoOsBbALlwMC+XEF+COtnO3iwsMyePduVL6/MZYJELnQkQUgaT2AYhmFsHxkJeEkEwXjiiSf8ZKOI4FvoRx99dNpPusLLL7/s7odEKgzDMIzixwTciAVP2vemU5GuG8UwDMMoWvJWwA3DMAzjv4wJuGEYhmHkISbghmEYhpGHmIAbhmEYRh5iAm4YhmEYeYgJuGEYhmHkISbg2wHvSDdo0MB9r9wwDMMwcskOFXC+LrZhwwY/OW/gM6LlypUzATcMwzByzg4VcL7xzTSZ33zzjb8qb2Ba0eIS8J9//tn9lclkDMMwDEPYYQLORCQI03vvvRc0bNgwq69+lSTSCTiTjmT6k4lW8OyPOOKI4LHHHnMToDDtaC6/vW4YhmGUfHaIgNN3vG7dunCZqTm/++47tUVq+Gwn+5QE0gn4CSecEM4axnSqTPepfzInOL8xY8aE+zFT20MPPeQMG6boLCoDB6OJb5czAYk/EQ33JdefRMVoyeR764ZhGEaUnAv4I4884qYJRbCY4pMJM5jqUk9DKeCJImoiXog202uy72effRbZ9u233w5Wr14dScsF6QScvDPdKHk+88wz/dUOZj1D3EXAuW66FzZu3Bj88ssv4bSr2wMefqsTT4yMOXjxxRfd3OLk8brrrnNTx7Zu3TpyL55//vli6eIgH9x7ri2d0cCsc76xkSkPPvigu3bNfffdFzvLnGEYRj6RcwGHVq1aOUEDPLCKFSu6+b99mjdv7gRGM2HChHBfH8Tm3nvv9ZOLlXQCDu+++27oZY8aNcpf7ViwYEEo4A888EDQsmVL9z9eeNL84tlAN4WeC11Tt27dsNzIB1PHAvO0L1myRG9apPTu3dsZZKk466yzYo27TME4ufrqqyNpGAx6/nPDMIx8JOcCToOKd6lFGKFmGU9U4PWsuBHqbFunTh0/OQSh+vLLL/3kYmOfffYJ7rrrLj+5AH379nXXGGeoCCLgeKYfffSR+799+/bB6aefntZLTQUe/UUXXeQnO8gXBpREOZgDnHMx93jHjh29rYsWygKDLImpU6c6Y2Z74fqZM16DMXPcccdF0gzDMPKJnAs4I6sRMsLmwpFHHulEXXj11Vdd/3AcvLb1zDPPuAb48MMPDx5++GF/k6BatWp+UomgQoUK7toXL17sr0qE8iqKcG+tWrUK9KMT3SA/77zzjivPbt26BZ9//rlbV6pUqXAUvEAXhhgWgNATXSgM5IVzf/LJJ8HWrVuDFi1aBF27dg3zyDWTB8RXw2C+jz/+2HUtCMxDTloqqE/+9WPsce2GYRj5SM4FfNGiRa7hpv8VT7lXr15OlJctWxZugzDfc889aq9/Wbt2rdv366+/dn21ffr0iQ2nk6YHyflceOGFQc+ePVP+imNgFf24GCoIE9eQSxA5vGsdRqcM99577+Daa68NRowYEaYTNtcGlYBI0mct92blypVuOwQ4W+ibLlOmTPDcc8+5bgX6+7lva9asceufeuopNwBQg5gvXbo06NKlizPSuEdELRipf8cdd0S29SHKMH369Eja008/7YwGwzCMfCTnAl6/fn3XUNMvyV+8Pl8sS5cuHfH0BITjgAMOCF87u+KKK2IFvEqVKsG4ceP85BC8dsKzqX5F4fXGgUCRZ4yWXEM5I4wjR44MBg4c6PJBOfnXesYZZySG+hmQd9RRR4XLxx9/vFqbOYcccojrCrnpppvCNPIj3R/UkyRxlUF+s2fPDhYuXOivjoWuF0bzazZv3uyMKd8zNwzDyAdyLuA0vIgw4M3xGpXu35Wwrh++BfosGcEu70Qfe+yxQdu2bb2t/k2nb7c48V8H41e5cmV/s1i4Pn47UjgI52MoxVGzZk13X+Jg8Bees3D55ZertZlDeTVp0iQsA4yqgw46KFyPAXHxxReHyz50wVSvXt1PToTt/W4ZXqXjPpSUVxINwzCyIacCTgiZBvOcc85xy3zEheXx48eH29CYkuaPPsdLJP2WW24J0zAACKv7NGvWzL0SlQSjrBk5nuqXTlxfeuml2F8mII5vvvmmn7zdfPHFF35SLIwupywp/zgQcF5ri4PXzUQIiZ5o44vX5PzX++LgE7qcX3vPiDevFQp4xpdcckm47HPllVe6Y8QN7psxY0YwZ86cSBr1wb8m+vs5RnG8JmcYhlHc5FTA27Vr5xpM/b42/c2k6RHovJrliyGviDHiW8LtCCyNPMI+dOjQyLaIAX2sSTC6Pd0vnYAXlnr16oUDxdKBkGYCIWXGFFCOOhzO//T3d+7c2YXOgVf4CIFr4WMsAaPOhUGDBsX2gQP3Ce+YCIfvucaJaRxEUgihC+zH+XhfW0adUy+S3pvH6//1l1/cWAlet+M65dU3YNnvlqHbpWrVqpE0BtBRpwzDMPKRnAk4XwDDc0NktNdMXzdpesTxueee68ReQ3+pDpfLgLAaNWoUEFuOl+tBYpkwYMCAoGzZsn5yIpkKIuBF+gLOGAGEDTFbv369G7B20kknhdswsI3R3353BeuTBBw4j28gYfSQru8Ffc777bef2upfCMHjJQt8hId9EXUxJBhl7r9NQP87gisG4KxZs9x+ug5I/7gepQ4YdTJATqD/n0/6GoZh5CM5E/BsQFBSCYhAw+/DIKhjjjnGT97h8GocnmuqiUl49x0Y1U03AAPNAE+REeD+79NPPw33jRNwwJDh06mM8gb6fR9//PFg2rRpwbZt2yLbangX3Y+CAF43A8x8EHTeKNBgHNDX7n8JLY64j+Eg6PoaEXe//OK+vsd35H3iBg1WqlQp7etnhmEYJZUSKeDw7LPPBoMHD/aT04KX5r87vKOhX5gBY8uXL/dXhSAk+qtkeK94tTBs2DBXFv5v9OjR4fZJAr498IU2OR795bzz3ahRI2+rf+GjL3EGVePGjf2kjGGUOP3ifoQlFfPnz3dfeNOQLz8i8+ijjxaIIhiGYeQTJVbAgUaWvu9M4Ytl2UyKUhTIiHg/DK1htDTvKvsD3vjxHjQDshjYpgUl6TWuJIpDwPHWO3Xq5ASUV7ro5kj6Jjkeru9pI+q+x5wtGA10qWQK/ftEHARC6ddcc43aIgjmzp3rwu+GYRj5TIkWcMj0S1+EmbPx1LYXzkUomRAv4W5GxJMHHwaEIayZ/MRLpG9adyHw2h3H93+HHXZYuE1xCDhwPD7+wvUmHRsjBoPjzjvv9FcVCZw36dw+GBmUuUDXhU+qrgPDMIx8ocQLeEkFQWNA2qWXXur+33fffQvM2U3olpHSmf4EPPLu3btHvk6XDr485wS8GL4glwmZjqwvbqxP2zCM/wom4IWEPmG8ZELnhImLcuAcXiNjADKFUeYvvPCCCwvHDeoyDMMwdj5MwAvJlClTgrFjx7r/+SiLDtsahmEYRnFjAl5IeCdd3ivmYyl8EzzVQDbDMAzDKEpMwIsABljlcgCdYRiGYZiAG4ZhGEYeYgJuGIZhGHmICbhhGIZh5CEm4IZhGIaRh5iAG4ZhGEYeYgJuGIZhGHmICbhhGIZh5CEm4IZhGIaRh5iAG4ZhGEYeYgJuGIZhGHmICbhhGIZh5CEm4IZhGIaRh5iAG4ZhGEYeYgJuGIZhGHmICbhhGIZh5CEm4IZhGIaRh5iAG4ZhGEYe8p8U8L/++itYvXp18PHHH/urHE2bNg3Kly8f/PHHH/6qkLJlywajR4/2k4uUkSNHBvvvv3/wxRdf+KsMwzCM/zhZCfhxxx0XHHzwwX5yWlauXBnsscceBX577713uM2ee+4ZWVe6dOmgdu3awVdffRVu07hxY7eObX2OOuqocN9UTJ48Odhnn32CXXbZxf0qVKgQrF27NrLN4Ycf7tb9/vvvkXQN65s1a+YnF5oZM2YEjz32WCSN43Oe1157LZJuGIZhGBkL+IcffujEpFy5cv6qtKxYsSIUTP3bfffdw2123XXXAuv57bXXXsGvv/7qtmnQoEGY/uWXX4b7/vzzz5F9kujUqVO4TevWrYMDDzzQ/c+533vvvXC7TAR848aNwZ9//uknF5oyZcoUyDsRgE8++SSSZhiGYRiQkYAjJLvttltw3XXXbZeAn3feef6qEETUP3b37t3dfuIhawFv3rx5uN306dPTCvinn37qrgEP/dtvvw3TFy9e7NL1fiLgv/32mwu1jxs3Lli1alXw999/h9usWbMm2LZtW7hMWP6dd94JJkyY4DxpzuezdevWYNasWcGiRYuCb775JkznWEQcOCf/f/DBBy59y5YtbhlD4ZdffnH/67A/5yRNRxAwbGbPnh2MHz/e5ZltBM7/2WefhcuGYRhG/pJWwPF+CXUjZM8991wBkc0EEfBKlSoFp59+evhbvnx5uE2cgNetW9ftJ6IjAl6lShW3vUDascceG+vFCm3atHHrNm/e7K8Kevfu7dY9+uijblkEnHPwv3jqRANEtFm+99573f+E+dmW9fSL9+vXz62vXLmyW4/wV61a1aWRR7oA+J9jwzHHHBMaEfXr1QvatWvn0i+//HKXxvERYhF5Qa5p7Nixbvmwww5zy126dHF5Y3vyxb0Dug78MjYMwzDyk7QCTti5Q4cO7v/tFXD/N2/evHAbCaEjgvwIr7N80kknhZ6vCLgcD8+UvmP+v++++4JSpUpFBE4jxkAcCCDr8PhBBHz48OHhNiLKePvA/yLgZ5xxhlvesGFDuD0hetIQ3iVLlrj/OS7XQr4ZSyDngzjjQws4XH/99W6ZY3BcykmW6Sfnf8YpCF9//bVLu/jii93yTz/9FIkaGIZhGPlLSgGn3xtvkYYftlfAMQToO5afHBcQcH54uyLel112mTrK/wSc/dj2pZdeckLI//RX+6FwzSmnnOLWIWo+/f5PKBngBiLguo+bc5LWp08ft8z/IuD8n/QjgjFx4kT3//z588Pj+WQi4EQPuNYXXnjBGT+sa9++vVtHXvxzy0+2MQzDMHYeUgo4Ys3rVPXq1XO/mjVrOi+X/7Mh2z7wpUuXuu0rVqwY6cPVAn7DDTe4sDmiXb16dbc+lYDjTbOuRYsWkXSOT2ifdd99951LEwHX/dTizV5zzTVumf9FwOlXZ3nhwoXBiy++GPnhHT/00ENu/dSpU8PjycA8IRMBB8LtvMLWsGFDt+7777936ffff79bvuCCCwrkgX5ywzAMY+cipYAzKEv/Jk2aFOy7777u/2wQAe/cubMTI/0T/D5w3sVmHwZjCVrARVD5vfHGG259KgFHqBFJzjNo0KAwrdWJJ7p9OJ8gAt6qVaswDSOBNMLhoAWc47HMADbguEceeaQTWQT83Xffdes5P6xbt87lFcNBQJTZhm3FkIgTcAmV89Ph8k2bNkXOAXfccUdQo0aN0PPnvfKhQ4eG6w3DMIz8JaWA+8SF0BEiPTo7jqQ+cP81Mn1sjsk2pEuoXQs4yHGEVAIO8iocPyIJ0u/Ox1IYcS6IgMt2clzC+xIRYFkEnBHi+pqkLx4BFpo0aeLSOJac99Zbbw3X49mTxjqMJIgTcJBz+e+HN2rUKDyHdEPwv+SZwYjp3pM3DMMw8oOsBPzNN98MOnbsGEmTEdOpwOOkD9r/MYpaYNCXf2z6pNnu7rvvdsuXXnqpW5bwM/vcfPPN4fYssz4V9GsTymfENgPbhg0b5m8S9OzZ0x2HV7JOPvlk90EZjo1QC1rAga+ltW3b1ol//fr1XbTCZ8SIEe6cfJCGfmyfIUOGOK+6V69ebpnX18iHeOQCeeFcuntBGDNmjOviqFOnjhvpr/NMCJ9jGoZhGPlPVgJu/A8E/PHHH/eTDcMwDCMnmIBnCR68vI+d7VgAwzAMwygqTMCzBAHn1bpRo0b5qwzDMAwjZ5iAG4ZhGEYeYgJuGIZhGHmICbhhGIZh5CElTsD151UNwzAMw4inxAg4c3oDnwploFjcdJxG7nn11VfdrGpPPfWUv8owDMPYgZQYAWeWL8Sbz5Gef/757rOrxo4Fo2q//fYL3nrrLTfzmZ5H3TAMw9ix5FzA8a7jhAAPj3m+mfiDKURlCsydEb7wlu7zsyUBZnvT8CW+TOErcTLRimEYhlH05FTAmbObD6D07dvXXxU29nwGFJFnco6dFb53zixhJZVZs2a5T8hyr7gfGFR8epYZ4OJgJjQ99Sphd75rz7fXfdhWuksMwzCMwpNTAQdE4eWXX/aTQ7Zs2eIn7XR89tlnJd4DHzBggLtXq1evdssff/yxW962bVtkOyaY4TvwPrVq1Yr9zjye+VlnnRX5RrthGIaRPTkVcGYDy2T2MmPHI3Ocy8QxeNgs60lYunXrFixbtixcFn744Qe37eeff+6vcvz+++9B1apV/WTDMAwjC3Iq4J06dXJTZSIGd955Z9CsWbNI6DVXcM6PPvoodqS7RAA2btwY6cP95ptvgvXr10em9uQ4bHfFFVe4+cmZGa1ixYpuxi+MlNdff90NzNOzgpHOgD2ZCIX9ZsyYEVxyySXOwGF7Znjb0R4qYX5mNBN+/PHHiAe+YcMGN7Atzhh75ZVX3P5M0cr85kcffXSBT8+eccYZwaOPPhpJMwzDMDInpwLOvNuDBg0KevToEaxdu9Y18gsWLPA3Swsh6M2bN6f8JQkgosJrUXiBTFOKkACeJWFf8kgffZkyZYIWLVq4dcOHD3dzc3/yySdOjBAuBJdzIFYI27XXXuv67Tk3y4gw4We2YR+2B6ZWxbudOHGiW2b7SpUqBdWrVw/mzp3rru3ggw92fc47Cpm/nYGEGClLly4NjjjiiMj0r0zt2r9/f7XX/2A/jJbbbrvNGSpMu8rxNG+//XaBueUNwzCMzMmZgDOnNYI9b9680MulUccTzhYEjn1T/RYuXOjv5hDPEQgPs63Mt/3www+7PBIC/uOPP9xf2H333cO5t/GSfTFimePq5TvuuCNcxgBApAXyLwIORx55ZNCwYcNw+ZxzzilwjlzSvn17d35EmvnN+Z/r1nA9M2fOjKQJGCgjR44MxzowD3nc9VDWlLNhGIaRPTkT8Hvuucc14jfeeGOYRn94YSBsm+6XCgnbjx071uVJwsK8yoYHrsEjZhsRc0DQNawnlC4gYDpkTHeB3idOwBs1ahQu9+zZs4DgYUDccsstkbRMef/9910U4LLLLgvTiCYAI8JbtmwZGjFAN4Ccn+34v2bNmuF6ypc0/zUzgXUXXHBBuNy1a9egefPmaot/iTMMDMMwjMzImYATjkbYBPqXDzjgALVF7pgze7YL3YsHnkrAgZA7IgeEvA866KDI+lwI+Jo1a4Jzzz03kpYJdBnQVQBbt24N6tat6waQ8S768uXLg/POO8+JKKPMAY+YvGrjCi8cb5lrBxHwJUuWhNsIRD7YF8MH2JbpV+O6SjgGoXTDMAwje3Ii4HiPvBM8adKkMI0BXwxiQmCyBW+QwXCpfs8//7y/m+OZZ55x3ihkKuCIGn3UQ4YMcR+a8fOcCwHv0qVLYsiakDsiW61atfD33nvvuXX0wfNeNtAnffbZZ7t++Pnz5zvDhOtZtGhReCzKgHMzTkGgfEjr0KFDmIYR8MQTT4TLAufW18K37dk37lUz0u3b94ZhGIUjJwLOSGQaa/0FNhp5vPCkj4OkgnAvx0r1S+pbrV27dnD88ce7/xlRTr7EW0wScAa3SR94HBxD9+WzjIEi3Hrrrc6DFTAwbr/99nAZ8SVfAgPAfAHHq+W6GKWOYAsMBMNYwRjhmuWnuxEQ2uuvv94ZHt27d3cCzvEYbMbocg0euS4TgUFs+hquvPLK2A/ylCpVyn2wR+A1M8YcvPbaa2qrwL1its8++0TSDMMwjMzJiYAziKl8+fIRUcHDkxHguQTPF4EiP/TLI0p49HiIvNfsixd5RoDYTn6EhOlHx8utUaOGE2SElWPgPbPMN8QZyIbHz/+k4UUThZD1V111lTNgJGpw3XXXBY888ogzIlju1atXmAfEbuDAgW5ZhG/06NHOu88UugH4iArivXjx4sg6vPBTTjklzAv94KNVFIGQO+mMjic/hPR9Aea1PLbRrwZibLGdHxHh2vnmvWEYhlE4ciLgJQ36ckVkeL871ac9WYco00dMny+veiGyepBWcYNQH3LIIS7PiKd84axt27ZBhQoV3Dr/x2t6wjvvvBOOLOeVLoS6KMAYoB89W4gEkG/DMAyj8PwnBTwbTjjhBPcBF5+bbrrJTyo2EGQZP8B71dIHP3ToUPfd8lTw3v2qVatcxIEPy9A/H/cBm8JAtwLRi6R37pM49dRT3QdsDMMwjMJjAp4G+ssJoSOg4oWfeOKJaV9VK0oI2cv5CIMz8IufDA6U0eE+vALGdnxkRkLt8r45H5YpimvgGPSFZ/pFPfKSreAbhmEYBTEBzwBGauOJ02fPx15yCXNxM0pdIJyvX71CxOmXLlu2bFC6dOnwRx8103/y7rf20pllLBvBNQzDMEomJuCGYRiGkYeYgBuGYRhGHmICbpQYeH99ypQpfrJhGIYRgwm4USK4+uqr3SBB+ut5ba4oBtgZhmHszJiAGyUCvtjGlK0PPvigfR/dMAwjA3aYgPuf8DT+O/ivkc2ePdtNTzpjxgw30p/pVw3DMIzU7BAB5xvkvPZEmHT16tUFvlvOJ0r5XreG1554paqkhFaZFKSw6LnBSzp8Q/3AAw/0k2NhdrlU87v37t3bfQ2Ob85r5KtxTDLD/dXfzM8E3oPnG+x+PcoFfMiGz+lmAp/h5ct/SfCpXboPtpfHHnssqFKlip9cLPBZ3riJarYXnnc++JMLnn76afc9BSO/oV3ldd9MaNWqVWSK5Hwl5wLOe8t8Z5tvhPOQ0qAj6AKTXPC1MCb40PAp0M6dO0fSdhSEeNN9AU1gUhP59CkQeSjsvN47gmOPPTYyi1wSfN2Ne5nq/XJmU2NWt6SJYVKJv6ZZs2aRGeH4Ol27du3UFtnBNK3jxo3zkzMCY4T51tPBe/npRIIZ6/hqXjZg8LRu3TqShnE5ZsyYSFpxgKHtT7pTVNAmcF8zhZnykj5olA7mMeCzxCWF5s2b+0k7PcyZ8Oyzz/rJWUEkT6ZFTgdzQciMjflMzgX8mmuuCYVYBFx/VpMJRZhvm8+HCjT4eC8l5fObiEUqodJgjBAaFvgsq0zvWdKRiVwyKXe+sZ7OU8dL5iHbHojM+ELI1+UeeOCBSFo2MHlMcXiRGr4bz6x2qcCbzTaKgHHVoEGDSBpf7svFOAImJOITvcXBk08+mbFBR+NP+/DDDz/4qzKCCXiI7pUUmOjov8aLL77oZq00siOnAo7o0Ugx9aUs62lG8U55EAcMGBCUKVMm3A8Pg7mrBRoOxIAvkPn9qRqm0aSxZ1s9BSfMmTPHeTysw6PzBXnp0qWuIWR93bp1I+t03jAuWE8F1NA1QDrXx5fUzjzzTJfOV9C2bt3qohAcu3HjxgW6BfDYJW+EnZNo2rSp+1Qq+Wdb8ey7du3qljGEfEhjHY2WH85FqAm9sp7JRhBL8v/777+H2zB9KevJPzOUCZ06dUr7fXj2A6IvTE/ql1k68JSZS13uyQcffODSySMPf7169dw6jD9/znYaRdZxfTIHOdEewvbsz/EoNx/qnh/yx2NbsGCB+7a8XzfII+fhx6Q3AsYNX/QT+Eqe1E2MViac0V/cW7RwofuiHut1OWvw1DgG94R8MD0v94rr4dooK+qRL05ES7i/HJt86QiYD9fObHJsy3Xr6AnRlPvvvz+44oorwmvmU8MC9Zp8kR5nCHJsKQPCmfrYnEvgvurnc9OmTaGXOm/ePGcYyT3ky4PAJDtybKJ5ug5raD/YVz+DzDHAfjzn4tVv2LDBHV/ng+du8uTJ4TJROT51DAzKlPNT133DjC8iso42RgwVIlRHHXWU64riXEldSbpd49jCcccd97+N/oFr5jhyf5nSV+pU1apVI+VNGdPm4XDwtUfYtm2bu8dsTx3wjVyeX/JPHbvxxhuD+vXrR+ZZ4BnheOzvG5maJk2auDrMs0g3EtDG84wQ6mZ/aasw7OTaZVpoeOmll8LPRDMBlZSf5F9H6DiWzg91bezYsa6Oci2+Z85MinJOplF+9NFHI9NF70hyKuDMc92iRYtwWQRcLGcGMI0aNcr9JDRHYVOoiD7f9qYCMu0mjTeVjgoUB/NaM+sWlRKPl4eCEc7Aw83DxxSar7zyimvwMQaAB5k80pgS0mE9jR0VDMQIAWYm4yF/4403/j2pYuXKlW6+bbYdP358OB82yxyfho58UcHFK+Uaq1Wr5mY6o/LSwHG9TEvqI+LKQ8u5CJmyjDdJvjAgqJBMmQobN2505UiDw/XT0HBuaZDIA5WU7anAjAjn3HK/yBtlSFiTBue0004L96fMOG8qj4lvr3MvmCIVsaJBkDLPFLxs8tinTx9XpjS+NIyUKQ8/fZl4nhgpGIGAkLMPZUIjzENOIwn0vWNYcX85nj/FKvBwM4e6gJEkfd7UZxofIEROedCQffjhh+6eYYxSlpQRZS8igciTRz7LS71AYDCIxFij0aJ+UgdeeOEF13D4cEzyzLXTb87/HJ9r4r6ddNJJwcsvv+w8f84t9/nuu+92os379kSDZO5534gE6hhG4sJ/jAnqE962XC+NP88ezwXz3XP95F9Hysgb50FweRbE8GVfni8afIw5nkMMK7prBHn+KR9/2loMXOoTEN2ijrZp08aVAdfEr3bt2q5OU4aIXFI7MWHChDCig2BRxzFGuZ6OHTuGBiyhWf4XIabB55i8/ggyLwFtFF0Y3FOcBJ5N8iLnR1R4Lnle2ZbphOVaEUSEhWhWUkSJ+sj1Ui60O5QpzwOQdw1CzNwDQNtKmSNGtDsIuNxLIA88j9xrHAyEEqOX9oLplemiYBsRfa4RAxEjlfaEfaVdpC5xb3nWaD+5Lo5FXYrjvvvuc8fm/lHuEnGlbjGgVQxo7iPXTjvC9VPetENAO8gcFXD77be74zGGgnrLNXA86Z7CiZLZELletuV+oikYhtpB4/nAeeReYiRQT2nreC5LAjkVcCqutipFwPGItmzZ4hoe0niopFLjFVERqRSIm+57RgRlOw3H4IbpPjEabzx8HkBukLY+tcFw2223uQZeN2hYxjTOwAAhtmVubh6mVKPpaYh1owT6IQAeIh4W4CGRB06ggsZZrxgjPLByLBoDjq29KRrq+fPnu2tBIHQonzS2p8GnvBAJv/+I9VibgEVLQy1I/yciSoPN/0nTsrKe8uP6KH9ElfMVpr+X/bQXxIPHAyVRHcD4wCsEwqt4QwIPou6ewTLnGEnQiNNQAd4F58fTBR5sGhKuibL2u0YYqEkDhPEg9YsGjXquIwQSeZJ8Iri6XsXNhgfUb7/+05dLYyb1gsaVbSgzygIh4a8g986PQAFGn06/ol8/59EAjTL7SRRE4DrEIJVyEzge0PjTGOtnjDxxPMqSOkVbATTY2tMC7jfGmkBjLNEcjslgJu3xYoj75STQyOMQANEw2gKBY3GvFv7jSfJcSf6oawgSXYE4CsD89hgRlDtips8vIgG0L36IXBu+iH2cMQWk+/3EDDIVA1NPE0x7R97JBwYPETd9XIwq6fLCyKHO6baZZ1NHfj785z5L20W7i1GlnzlEX4QP48pvs2hHMBriIF/SvgLPF+eiTRN4XvDo9TVgfIvXTZlKe8J99M/Pc4uYA4aojHnBsKFsBJ4XuQ6cGfKhozc8l6QR4SoJ5EzAKXz/IRIBpyLwEGJ5wrRp01w6NxKrjoqFOJCGaOG98OOm+ccUGjVq5CoFjSwWnkAjyT5yDH5URqlAHJMbqNdzg3kYgNAXlYNjJDWsAvvqB46Krx9eqbgyCptjsl6fm/Vi5WsQHj24DG/O7xvmoeSBw1qkHP1GmvNhOOE5xY3IZD1eGMfgfyxiyRf/k0YDgddLNCQJxJTGXB4+ujP8vGZCXB2iEWJchQZPA8NFwHLGECIfhMAkzArc11SRA23c4VWIQYPIIFY0LCtWrHDbkBd971iPYHO9EsKjkZVjaPR18Sxw72jMeRaSIMogQidwn7V3wP7i+ZAPjqvzSJ45tzYqNXSLUHaIAAbriBEjXDp1kmfMR64ZpM4QUZIoG3UAI8LvvqGRFC+Vc4iHxPPgixbHxVvWyyI2/PWf71TtBPUSoZD76RvktA14hdL+kE+eu7vuusuJNveTNO4D1yiOSNL5edY5D2mUqxYpX8jiwOBCtBEvvHg8QjxX4F5JmJv7iqcK5IH86TxJmB7wMolgaagPdI9gEBI94Z5I24VB5Q9sxUsnOsU18GxjvOrzsZzURhBt0MYeBiDtrA8OBhELBjRT/lyjXDtlKu0LBp7fLaTXU8ZiYGGQSjkB3rkY+DiM06dPD9cB+0ndLAnkTMD79+8fhlgEEXBClzpMiJdEOhaQhMoIi/CKAJaP/ukH2YdQJg8ZlVcqKBY3x/SPw8MnXgD7+eslLMgDQwXlJlOJkuDauF7dMBLWxDoUpNGgccB6pOL55+XnCy8VkWPTsAjnn39+RMhoxKXR4IEndK3BApfGn4pKZEFDRaWik38eEowYP1/ykBAhGTx4cGR/QfpkxcsVo8z39jOBvmLpYxTIo24EgfNxfYA3xkNK/zOeCutEYOR+S594HHjVbCNhXmkEMBC4Dpbx+BEpv3z4AULMcaS++4Yf9Yh6oKF8OCaC6/cdCxxXv+FAXvzj4GEQ0QLOjaHn51HyqeH+02BzbwlBUnfZX7x3IksYRhqJrAjkBxHAiJZ9pTx9g4H7Q7cB0ODz3LKNlLFAWehQMfWL6JKAiJz2j+j41xfXTiB2cv+pE36IF69SniHgf87H80/50FeOKOHNYRgCXSg8y+nOT1RPxl9IFEOiJUng7WNQ0H5xT7kHbI/RANRRHBTaTenmYR3bEK3x8yT3kvusn3+MGO4jIWlC5LQr0s8NHE+cDoFni/aMNon1PJNJ5/Ph+eTZFjCUdbQPMCIwAnA2Jk6cGDpRUjdEW4jykK4jXIiwlCvjVmQ/HEdpfwXqFlrDeum61ZAPDJ6SQs4EnH4/3ecC0qDxgOt+ZBEfClBEgoKT/g6BkLDfEABehw4HYcFLWITwloTMgIeYgTAgYTI9mpUHjxAoSLiRikqDw8MUd36gr1hXMCAPeuAW/WPixXAtNFwavF/CWz40IgiXDu1gwepRnAwCIWQNVHr93jp5ptHDqAL2lTIQsLIldImnJ2MAgGuijMWK1V6XDyKERyzlROMjfVXZwr3Tg6TkYdVQJtLAMwc6RpEg90+8NcrfH4TmwzVyfTQe2lCg31GMS8rZjwLQQNNoAftjuNCocH4xLgDxII+UE+eiz1g3PogDDXsceHraY+SexHnkUi84t/ZmuSc0bnEDvHhGZDATUD/ZX7pAuCYEQ4MA4uWzXg/uAkQPYSP0zXG0USpdCFKH8MR57jCS/PtLHdBGMHVYvwJGvaetEXiWuRf6ORToD5Y3AzBsdNiV7bke7ovAM8cyYXAgskc4ludaulUwpOW+A+l0+3E8vFT/9VPOzzgAwCBIejZwKigjHZrH8KONFKhLRDHJjxil3Htf0BAs6U7jeL5jhUEvni1wLARdIlUcT997eTOEaxQDTdcpDCG/fdGwr34Vk/31d0DEkNIDlok2SN2gvOX5o12uU6dOuB3QxsqbTzyz0q7RtcEzpOsG5UlZSTesdpKov6TJWKqSQM4EnD5jfyCJCDgPji5EsURFgACvhYo2depUVzlo2DhmHOx74YUXugaK8AgGgvSZ0UBSYXiwuDmIl8zxLV4nDQANCDeMB1QqM5EBrFXyioHAzRTr12fQoEEuHzycPLhi7WkIv7EdSOPO6Eb+p/GiMY4TcIwA+jkF6ZeRRgTwzmRfGg3WU2ZEO2g0tDGFJc+9oawwpFjm3GJxyzveCCLXwrkJY8k94+HWDYsgI3wZtAYSiuc86Uasx0HDhCEig1rwevxX1xg8Jn2BjCDnXtPw4Elw7/QHTrgeDBNC80mGGFAn5D4JNHIYAEB+KAMaReomEQ/OK5a99D/yo77haSLiCIiM+qcREg8W44BtEVttOGmof5Ql+SL/gKFEn72G+ypiSQice0+jjrFKOSF+ceKGZ0k9IE8y0Ev6G/HmWKaO0JDjjREdwxDhXNx3nlUMHgxkPEcMIEBQuUYG19EwY3RQPv369XPrxYMDicT9//bOtbuma43jn8AYxvDCWwPDG1+CF76AN6cl4pq4RKjUcBycxuUcdWguJdSlCaU4jVOt41KHkFONFlWaCkpcQlziEuEUyUg7xuz+zfbZnXvtnRD22Wz+vzGesdee6zb33GvN/3zmnGs9nIPyQtz4Th1gMFkMD45Gtg3ncHyOzT4IflfXGmJsQwJWnogo+WcyE+UZNjSo7C1vwGxwvvN/G9w/lCn/5+XLl/3QFMcEvD+uJfJIudJbYOPUwHwfumdtTkwI9QjnonuZ64rhIP6DaHcu90j49AP5J408USbUrTgK5sSYoxFCPcVscOosnCnqQLaxRgGOFM4L5cdxEUQbEjPvnfqXMqXBzf8b/mdRuFYKCwt9/UXZR4cRbMIujQC7v9jGeuP4vTa/gnopHDIFGvS2nvlN3HfAMFA4nBadUzV48OB4o435TFx/lE10mOVFkjEBtwswxCp0q+ANazVGPQPGJ0jHuDFSVTxABcNFQWGzbdjVxz7WHcg20UeZ6ILiT2M9lV+4L+cPx6Pp6qGLMhV0F3FzkgfOScsw+vujnqtVCBiVY7Sr1cAzMO8Z6G4NZ3TbeJ31QnB+RIE0zskFHpYdFQg3Nev57dwwlE3YTcbsZcsbFZLtbzd4KrhZQoHlPOxvXbo9pbi42O9vjQ8q2OgENCore4kJlRf/IftQofAbwscJqTBYT5l0J+DmXRhUbBzTeodYR8Vh5UPlb+LNtuGjgHSXcjy2M6+eZRMKBMCuXfIW7aoMYZyQfa0MaKBEtw8nUZIXhhQsn4h/V7+bsqEiZjuuNcrNPA+8azwZGlB2LLz1UOyOHjkSX4dXHK7jmrF7k+vcntAAjmmiRLlyXbMd9xLXZTibH8w7iz5mShrnQKC6qic4Vjj+ifdqecaji5YNQhTerwhBqmvHhJ510YmjdBXbOayBZNB4ocyj467AvtZlznHp8ua/tKdMDCaKRX8vjgjXFPtyf4dPW1B/2SOuBg1eOw/3MOUejpHze7km6L3iXsN7ZnsDz9bOx/0YfYwxCo05258GVdhrBvwea7xhdKFT93IdAb0eJqqUX+g125wI6h70hN9kQ1IMM4T/P0Id9sKwD9c5v5O6mUZZtLfiRZMxAQdaQlTCQmQDVK54eM/78hkhXhXo4QnHq/GYaaSHT3q8CtCoCSfq4RTSMAsbBy8DGRVwxnGirUMhXkaolLhhox6+EK8zeNN4sYwr0xNGD8qzvob4ZcZ6TBku5HfSc8GQzstGRgVciGyBIZDuZqcL8brC8ApDE8z9eJnGg9MNdQDj68zB6O6Nny8SCbgQQgiRhUjAhRBCiCxEAi6EEEJkIRJwIYQQIguRgAcwQf7ho3Z3+8592XPanbsPXHv7H8+4CiGESC8S8N9p7+hMEiHZ8xtC3vlz4rvchRBCPD+vjYAT1rGrN4A9etyRJDyy9Novv6R+45cQQohn47URcN47HL7uz+AVj1GxeZF263ZbUtqrYkIIIdLHayPgvAGOdx9HQ3O23vtfktCEdr6xye3dV+t27t7nfjx3KWn9kwxBfmPk6KT0VLZ0WbkbNXpCj87TeOGKz9/uL2rchYtXk9Y/yZqu3HB/ejM3LQ2Hsz9edCe/P52Ubqa38AkhRPp4IQJOkI0wfGgUol8R9SqEyp8oPiFEg2pra0tIC+HVrWGELl6sb6H7gGNGRSa0o8dOupGjxrmSshWutKzCvTlyjKv98nDSdt1ZTwScc7XcupeU3pX9e+de98aI0e69kuXu3X+U+OWDtXVJ23Vn6RLwbf/a4ctnWSwv0XVm99r+CNMqhBDi+ci4gPfv3z8eichCDAKh3IiIRYQY1pWVlXlvmXfQEhGGKDCEDYT6+nofzs6i04SRcoiURZQqItqwLgzHSWhGIlIZjx51P/adkzvOHTl6Iv695sAht6JijV++eKk5Jpqlbu2HHyWIHwI6Z94Cvy3fowJeWbXJf65Zt8HNnlPsTvzusVat/9iNyBnrPqzc6K7fuOPT1sa2WV6x2jVfu+W/Hzv+vav7+pibOWuOP27umDx35uyF+LF3xAS9pHRF/PvefQfdrNl/TRD1w19/6+a9s8iLP99DAb/a3BI/P970vHcW+nzZvl3ZzZZWN+Pt2a60fGW3Ao4JIYRIDxkTcEKzIdiEdTQQcptYRrg4RNpiLIOFtSM2rUF4T7az6DcWys4CwhNSkRfsp4rTjQcevpD+Sd3nObnjE0TKjO+I7Vd1R73niZCStnrteldQWOR+aDjn3iqa5Xbt2R8X8Bs37/ru8Z27/uP+tniZK1++ym83emy+F+hPqj/zHvjWT7b7bfFmV65a5zZs3OI9a867/fNdfnnP3gOu4cx5v2yNh2j+Vqxc6/4yd74X4sLpM13Nwa98Osv1P5z1+Vu8pCQu4JyTBkvd4WO/5Tl2bBoXxQsWu81btiWVze4v9rvPPt+dkLY81riRgAshRGbImIATrJ4Yrq2trXFDiMvLy/16xLW0tDRhH4LA403b2CmNADxrPPAQGgImzNEYvyHRMfCouERt3ITJ8WXEHEPsbt6656bPmOXWxbxVjLQrV2+6CfkF3rMmbcnSMjdpyrS4GOZNLPACz7EqY17tlKkz3KbN1d6Tt3NYQwARDRsP84oXuY+3VHsBx+sn7cTJBt8gYBmPOcwfaWPHT3IfrK70eWGfBQvfdZcuX/NDAaTRc0DjwQSccy8reT9+Thoo5e+vcodijZSwTLCNm/7pxX/R35e6j2LL167f9ukScCGEyBwZEXDGofGS6+rqvPiGhigTcD7VDPHevXsnBJ/funVrQne5gYATu7mysjLlcSA/P997+SH3HzxMEpjQpk5723vG9t3EGi93fkwQLf3T7Tv9J6J8/sIVv/zt8XrXcPp8XMDNAz/438N+0hmix3g33ejfnTjl9zEBvxgTWhNiLH9Sofvy0DdewPGsLX3SlOlu26c74t8XL3kvvh89AdY4QOyZGEdeag7+1rV/uem6++bId3EB5zfh9TPuz34IM9vRy0D+7RwYx9m1Z18sP7sThg8k4EIIkTkyIuDNzc1eWMPQc3jViDcUFRW5gQMHxtcZ7BNOQispKXHDhg0LtnAuJyfHDRgwwC8PHTrU5ebmJqwHztWvXz/X0tKSkN7Z+XOSwITGDO9Ro8d7AcPwUEfkjPEe+OSC6d7LRfTmL1zst6drmu8IK5+XYiIZjoEj6HjJH6yp8sd7q+jPvtsc7531JuAY3d8cg/V5E6f6tKiAN5xp9Ostf3Zc1uE5sz9d5nyamLM8cfI0vx3d6+EY+PFYQ2LCxAK/TD7pUWD79Rs2J5XN/gOH3J69NQlpEnAhhMgcGRHwzs5O16dPHzd37lw/A53xaQKl2zg1XnV1dXXCPo2NjW7QoEEJabW1tV7UmUnOMQsKClzfvn1dR0eHX88kuKampoR9gIlrQ4YMSfkYU1RgUhleNR50NL3x4tWELnCzU6fPxceiuzLWn2o4l5QeGl3jTCyLpkeNx8dS5Q+jUUGDI0yjIfE0s93x2rs67rNYe3tntPiFEEI8IxkRcKipqfFj0Ahwr169XEVFhU9nTJr00NOG4cOHewtBgGfOnBmffZ6Xl+ceP37s1/HoGWkIe3QfJrUx5p6Khw/17vNMmRBCiPSRMQEHxBpvOfoylZ5C1ztj3k8Dnn9VVVU0OYG7rQ+SxEaWXmO+gRBCiPSRUQHPNIyJM77+NEjE/3+mF7gIIUT6eaUFvKc8fMKLXWQ9NwLFCCGESD8S8AiMmXd0dLq2+z/5UJhRQZJ1b5QZ3eXM8E8xZ1AIIUSakIALIYQQWYgEXAghhMhCJOBCCCFEFiIBF0IIIbIQCbgQQgiRhUjAhRBCiCxEAi6EEEJkIRJwIYQQIgv5FU2VLrnahWX7AAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,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>