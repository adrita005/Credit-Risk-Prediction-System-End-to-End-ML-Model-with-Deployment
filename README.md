# Credit-Risk-Prediction-System-End-to-End-ML-Model-with-Deployment
Problem Statement:
Banks and lending companies lose a lot of money when borrowers fail to repay their loans. The goal of this project was to build a machine learning tool that flags whether an applicant is high-risk or low-risk before a loan gets approved, helping credit teams make faster, safer underwriting decisions.  
Dataset Explanation:
The dataset contains a practical mix of financial details and applicant demographics. It includes information like monthly income, requested loan amount, loan term, credit score, collateral value, past defaults, age, and employment status, all tied to a target label showing if the person defaulted or not.  
Preprocessing Steps:
To get the raw data ready for modeling, I engineered domain-specific features like Debt-to-Income (DTI) and obligation ratios, and grouped ages into clear categories. Then, I used RobustScaler to handle numerical outliers, One-Hot Encoding for categorical text columns, and applied SMOTE on the training data so the model wouldn't ignore rare default cases.  Visualizations:
I plotted correlation heatmaps, boxplots, and distributions to see what drives loan defaults. The charts made it obvious that applicants with low credit scores, high DTI ratios, and past defaults were clustered heavily in the high-risk group.  Algorithms Used
I tested four different algorithms to see what worked best: Logistic Regression as a baseline, Random Forest, Gradient Boosting, and XGBoost. I then used RandomizedSearchCV on XGBoost to fine-tune settings like tree depth, subsample ratios, and learning rates. 
Performance Comparison:
While all models performed decently, XGBoost came out on top with the highest ROC-AUC (0.92) and an F1-score of 0.88. It struck the best balance between precision and recall, ensuring we catch as many risky borrowers as possible without unnecessarily turning away good customers.  
Deployment Details:
For the final delivery, I packaged the trained pipeline into .joblib files and built a simple, interactive Streamlit web app. Credit officers can type in an applicant's financial details and instantly get a High/Low Risk assessment along with calculated indicators like DTI.  
Learning Outcomes and Challenges Faced:
The biggest takeaway was learning how to handle class imbalance properly without causing data leakage. In credit risk, catching risky borrowers (high recall) matters much more than simple overall accuracy. Tuning decision thresholds and turning a raw notebook model into a working web app gave me great end-to-end practical experience. 
