# Graduate Applicant Screening: Predictive Modelling & Fairness Audit

## Overview

This project develops and evaluates a predictive analytics workflow for screening graduate and early-career applicants at a mid-sized Australian technology company. Using **Orange Data Mining** and Python, the project covers data preparation, neural network tuning, model comparison, SHAP explainability, and an ethical review of automated recruitment decisions.

The objective is to assess not only how accurately the models predict job offers, but also whether their predictions are explainable, fair, and appropriate for recruitment use.

## Objectives

- Clean and prepare applicant data for machine learning.
- Tune neural network hyperparameters using grid search and cross-validation.
- Compare a neural network with a decision tree.
- Interpret model predictions using SHAP.
- Identify potential gender bias, proxy variables, and privacy concerns.
- Recommend safeguards for responsible AI-assisted recruitment.

## Tools & Techniques

- **Orange Data Mining** — data preparation, sampling, model evaluation, and model explanation.
- **Python** — neural network hyperparameter tuning.
- **Data preprocessing** — missing-value imputation, categorical label standardisation, one-hot encoding, and numerical normalisation.
- **Machine learning** — neural network and decision tree classification.
- **Model evaluation** — stratified three-fold cross-validation, precision, recall, F1-score, classification accuracy, and AUC.
- **Explainable AI** — SHAP-based feature impact analysis.
- **Responsible AI** — fairness, privacy, data minimisation, and human oversight assessment.

## Dataset & Preparation

The dataset contains **3,000 applicants and 14 original columns**, with `OfferMade` as the target variable.

Key preparation steps included:

- Standardising inconsistent `GenderIdentity` categories.
- Converting the target labels from `0/1` to `No/Yes`.
- Imputing 300 missing values in `CareerBreakMonths` using a model-based tree imputer.
- Removing `Applicant ID` from model inputs.
- Applying one-hot encoding to categorical features.
- Normalising numeric features to a 0–1 range.

The prepared dataset contained 18 feature columns and one target variable.

## Model Development & Results

A grid search tested 15 neural network hidden-layer sizes using three-fold cross-validation. The best reported configuration used **10 hidden nodes**, ReLU activation, the Adam optimiser, and a maximum of 300 iterations, achieving an F1-score of approximately **0.76** during tuning.

The neural network was then compared with a decision tree using stratified three-fold cross-validation.

| Metric | Neural Network | Decision Tree |
|---|---:|---:|
| Classification Accuracy (CA) | 0.828 | 0.788 |
| AUC | 0.899 | 0.840 |
| Precision | 0.749 | 0.688 |
| Recall | 0.730 | 0.673 |
| F1-score | 0.740 | 0.681 |

*Results are approximate and reflect the reported model-comparison experiment.*

The neural network outperformed the decision tree across all reported metrics. However, its greater predictive performance comes with reduced interpretability compared with the decision tree.

## Key Findings

### 1. Job-related predictors

`ProgrammingAssessment` was the strongest reported feature influencing neural network predictions. Higher programming scores generally increased the predicted likelihood of an offer, indicating that the model captured a potentially job-relevant signal.

### 2. Gender bias

SHAP analysis identified `GenderIdentity` as one of the most influential features. Male-coded values pushed predictions towards an offer, while female-coded values pushed predictions away from an offer.

The decision tree also relied on gender, indicating that this concern was not limited to one modelling approach.

### 3. Potential proxy variables

Other features may indirectly reflect sensitive or socially patterned characteristics:

- `InstitutionRank` may reflect differences in educational or social opportunities.
- `CareerBreakMonths` may be associated with caring responsibilities.
- `WorkExperienceYrs` may be associated with age and may disadvantage applicants with limited experience.

Removing gender alone did not fully eliminate the predicted gap, highlighting the need to investigate correlated features and historical recruitment labels.

### 4. Graduate recruitment suitability

The model is intended to screen graduate and early-career applicants, yet limited work experience can reduce predicted offer likelihood. This raises a concern about whether the model's learned patterns align with the intended applicant population.

## Ethical & Privacy Audit

The project applies the SCORE framework to examine responsible use:

- **Stakeholders:** Applicants and employers are affected by automated screening decisions.
- **Consent:** The case does not establish whether applicants are informed about automated processing or how they can challenge a decision.
- **Outcomes:** Predictive performance alone does not establish fairness. Outcomes should be assessed across relevant groups.
- **Rights & Rules:** The audit raises concerns about data minimisation under Australian Privacy Principle (APP) 3 and data quality under APP 10.
- **Escalation:** Clear accountability, human review, and an appeal process are needed.

## Recommendations

1. Do not deploy the current model for live recruitment screening.
2. Remove unnecessary and protected features, including gender and age, from model inputs.
3. Investigate proxy variables and review historical offer labels for potential bias.
4. Evaluate selection outcomes, false positives, and false negatives across relevant groups.
5. Inform applicants when automated screening is used and provide a way to request human review.
6. Establish a named model owner, documented escalation procedures, and meaningful human oversight.
7. Monitor model performance and fairness regularly after any future approval for use.

## Conclusion

The neural network achieved better predictive performance than the decision tree, but the fairness audit revealed significant concerns about gender influence and potential proxy discrimination. This project demonstrates why responsible machine learning requires more than predictive accuracy: models must also be explainable, privacy-conscious, and evaluated for their effects on people.

The recommendation is **not to deploy the model as it currently stands**. Further fairness testing, feature review, historical-label investigation, and human oversight are necessary before operational use.

## References

Kingma, D. P., & Ba, J. (2015). *Adam: A method for stochastic optimization*. 3rd International Conference on Learning Representations (ICLR 2015). https://arxiv.org/abs/1412.6980

Nair, V., & Hinton, G. E. (2010). Rectified linear units improve restricted Boltzmann machines. In *Proceedings of the 27th International Conference on Machine Learning (ICML-10)* (pp. 807–814).
