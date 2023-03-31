<h1 align="center">Customer-Churn-Prediction</h1>

<h3>Introduction</h3>
Customer churn is when customers stop using a company's products or services, and it can be voluntary or involuntary. It is important for businesses to track customer churn because it can be costly to acquire new customers. Retaining customers can reduce churn and improve a company's bottom line.
<h3>About the dataset</h3>
The dataset -1 contains customer records associated with a European Telecom Company. The dataset -2 appears to contain information about customer support calls, with each row representing a single call.From the features provided, we can potentially analyze customer churn rate and service performance of the telecom company.The dataset is taken from:-
<br/>
https://www.kaggle.com/datasets/datazng/telecom-company-churn-rate-call-center-data
<br/>
<h3>Research Questions</h3>
Dataset-1
<br/>
Q1.What is the overall churn rate of the telecom company and What are the factors affecting?
<br/>
<br/>
Dataset-2
<br/>
Q1.What is the overall satisfaction rate of customers from customer support?
<br/>
<br/>
<h3>Data Cleaning</h3>
To ensure that our dataset was of good quality, we first explored the data to understand its columns and contents. The dataset contained 7043 rows and 23 columns of numerical and categorical data. Dataset -2 contains 10 columns and 4054 rows. We identified missing values and duplicates and removed them from the dataset. We also performed some preprocessing on Dataset-1, including handling outliers, dealing with categorical variables, and normalizing or scaling numerical data, depending on our research questions and the dataset's specific characteristics.
<h3>Exploratory data analysis</h3>
<h4>Dataset-1 -Customer data</h4>
<p>
 <img src="https://github.com/isabeljohnson001/Customer-Churn-Prediction/blob/8b727b5c505f8e8216d6ae558d66f75d1202f7fa/seniorciti.png" width="500"/>
 <img src="https://github.com/isabeljohnson001/Customer-Churn-Prediction/blob/8b727b5c505f8e8216d6ae558d66f75d1202f7fa/gender.png" width="300"/>
 <img src="https://github.com/isabeljohnson001/Customer-Churn-Prediction/blob/8b727b5c505f8e8216d6ae558d66f75d1202f7fa/Conrtact.png" width="500"/>
 <img src="https://github.com/isabeljohnson001/Customer-Churn-Prediction/blob/8b727b5c505f8e8216d6ae558d66f75d1202f7fa/Servi.png" width="500"/>
 <img src="https://github.com/isabeljohnson001/Customer-Churn-Prediction/blob/8b727b5c505f8e8216d6ae558d66f75d1202f7fa/PhoneServi.png" width="500"/>
 <img src="https://github.com/isabeljohnson001/Customer-Churn-Prediction/blob/8b727b5c505f8e8216d6ae558d66f75d1202f7fa/Internet.png" width="500"/>
 <img src="https://github.com/isabeljohnson001/Customer-Churn-Prediction/blob/8b727b5c505f8e8216d6ae558d66f75d1202f7fa/payement.png" width="500"/>
 <img src="https://github.com/isabeljohnson001/Customer-Churn-Prediction/blob/8b727b5c505f8e8216d6ae558d66f75d1202f7fa/payement_churn.png" width="500"/>
 <img src="https://github.com/isabeljohnson001/Customer-Churn-Prediction/blob/8b727b5c505f8e8216d6ae558d66f75d1202f7fa/partners.png" width="500"/>
 <img src="https://github.com/isabeljohnson001/Customer-Churn-Prediction/blob/8b727b5c505f8e8216d6ae558d66f75d1202f7fa/paperlessbil.png" width="500"/>
 <img src="https://github.com/isabeljohnson001/Customer-Churn-Prediction/blob/8b727b5c505f8e8216d6ae558d66f75d1202f7fa/onlisecu.png" width="500"/>
 <img src="https://github.com/isabeljohnson001/Customer-Churn-Prediction/blob/8b727b5c505f8e8216d6ae558d66f75d1202f7fa/depend.png"width="500"/>
 <img src="https://github.com/isabeljohnson001/Customer-Churn-Prediction/blob/8b727b5c505f8e8216d6ae558d66f75d1202f7fa/Tenure.png" width="500"/>
 <img src="https://github.com/isabeljohnson001/Customer-Churn-Prediction/blob/8b727b5c505f8e8216d6ae558d66f75d1202f7fa/TechSupport.png" width="500"/>
 <img src="https://github.com/isabeljohnson001/Customer-Churn-Prediction/blob/8b727b5c505f8e8216d6ae558d66f75d1202f7fa/MonthlyCharg.png" width="500"/>
 </p>
 <b>Observations:-</b>
&emsp;<li>Contract type: Customers on a month-to-month contract have a much higher churn rate (42.71%) compared to those on a one-year (2.85%) or two-year (11.28%) contract.
<br/>
&emsp;<li>Tenure: Customers who have been with the company for a shorter period of time are more likely to churn. For example, customers with a tenure of less than a year have a churn rate of 38.78%, while those with a tenure of more than five years have a churn rate of 6.71%.
<br/>
&emsp;<li>Seniority: Senior citizens are more likely to churn (41.68%) compared to non-senior customers (23.65%).
<br/>
&emsp;<li>Internet Service: Customers with fiber optic internet service are more likely to churn (41.89%) compared to those with DSL (18.99%) or no internet service (7.43%).
<br/>
&emsp;<li>Online Security: Customers without online security are more likely to churn (31.37%) compared to those with online security (14.64%).
<br/>
&emsp;<li>Tech Support: Customers without tech support are more likely to churn (31.23%) compared to those with tech support (15.20%).
<br/>
&emsp;<li>Payment Method: Customers who pay with electronic check are more likely to churn (45.29%) compared to those who pay with bank transfer (19.20%), mailed check (15.25%), or credit card (15.75%).
<br/>
&emsp;<li>Monthly Charges: Customers with higher monthly charges are more likely to churn, with a few specific charges showing a 100% churn rate.
<br/>
&emsp;<li>Number of tickets: Customers who have opened more technical or administrative support tickets are slightly more likely to churn.
<br/>
&emsp;<li>The three features that are most positively correlated with churn are numTechTickets, Senior Citizen, Partner, and Online Backup.
<br/>
&emsp;<li>The features that are most negatively correlated with churn are Contact, Payment Method, PaperlessBilling, OnlineSecurity, and TechSupport.
<br/>
<h4>Dataset-2 -Call Center data</h4>
<p>
 <img src="https://github.com/isabeljohnson001/Customer-Churn-Prediction/blob/8b727b5c505f8e8216d6ae558d66f75d1202f7fa/Agents.png" width="500"/>
 <br/>
 <img src="https://github.com/isabeljohnson001/Customer-Churn-Prediction/blob/8b727b5c505f8e8216d6ae558d66f75d1202f7fa/AdminSupport.png" width="500"/>
 <img src="https://github.com/isabeljohnson001/Customer-Churn-Prediction/blob/8b727b5c505f8e8216d6ae558d66f75d1202f7fa/TechSupportPerfo.png" width="500"/>
 <br/>
 </p>
 <b>Observations:-</b>
 &emsp;<li>The overall satisfaction rate of customers was 3.40, and customers were generally satisfied with all topics.
<br/>
 &emsp;<li>The topic with the highest resolution rate was "Streaming," while the topic with the lowest resolution rate was "Contract related." .Technical Support had the highest number of unresolved cases, followed by Payment related, Streaming, Contract related, and Admin Support.
<br/>
 &emsp;<li>The average speed of answer for different topics and agents varied, with Martha being the fastest agent for Admin Support and Streaming, Joe for Contract related and Technical Support, and Greg for Payment related.
<br/>
 &emsp;<li>The agents with the highest resolution rates were Stewart, Greg, Becky, Joe, and Dan, while the agents with the lowest resolution rates were Diane, Martha, Jim, Stewart (appearing on both lists), and Greg.
<br/>
 &emsp;<li>The overall satisfaction rating for calls related to Admin Support was 3.43, while the overall satisfaction rating for calls related to Tech Support was 3.41.
<br/>
<h3>Evaluation on the performance of machine learning models</h3>
 <img src="https://github.com/isabeljohnson001/Customer-Churn-Prediction/blob/42207c64f31c5257bff6db02138d7ac96cec9c63/Logi.png" width="500"/>
 <br/>
 <b>Observation:-</b>
&emsp;<li>Based on the evaluation results, the best performing models in terms of accuracy are the Random Forest Classifier and the Logistic Regression Classifier with accuracy scores of 0.86.
&emsp;<li>Based on the precision, recall and F1-score, the best model is Logistic Regression, with a precision of 0.75 and recall of 0.72 for identifying the churned customers. 
&emsp;<li>The Random Forest Classifier also performed well, with a precision of 0.79 and recall of 0.64.
&emsp;<li>The Logistic Regression model has a higher AUC-ROC score compared to the Random Forest Classifier, indicating that the Logistic Regression model has a better overall performance in terms of correctly identifying true positive and true negative cases.
<br/>
<h3>Recommendations</h3>
<b>Based on the insights provided by the two datasets, here are some recommendations:</b>
</br>
&emsp;<li>Contract type: The company should consider offering longer-term contracts to customers to reduce the churn rate. One-year and two-year contracts have significantly lower churn rates than month-to-month contracts.
</br>
&emsp;<li>Tenure: The company should focus on retaining customers who have been with them for less than a year as they have the highest churn rate. Offering discounts, promotions, or personalized offers to new customers could encourage them to stay longer.
</br>
&emsp;<li>Seniority: The company should look into the reasons why senior citizens are more likely to churn and address any specific concerns they may have. They could also offer senior discounts or other promotions to retain them as customers.
</br>
&emsp;<li>Internet Service: The company should consider improving the quality of their fiber optic internet service or offering incentives for customers to switch to DSL or no internet service, as customers with fiber optic service have a significantly higher churn rate.
</br>
&emsp;<li>Online Security and Tech Support: The company should consider offering online security and tech support to all customers, as these features have a negative correlation with churn. Customers who do not have these features are more likely to churn.
</br>
&emsp;<li>Payment Method: The company should consider offering more flexible payment methods, such as bank transfer or credit card, to reduce the churn rate. Customers who pay with electronic checks have the highest churn rate.
</br>
&emsp;<li>Resolving customer issues: The company should focus on resolving customer issues more efficiently and effectively, as unresolved cases have a negative impact on customer satisfaction and retention. The company should identify the reasons why certain topics have a higher unresolved rate and take steps to address them.
</br>
&emsp;<li>Agents' performance: The company should consider providing additional training or support to agents with lower resolution rates, as this can improve customer satisfaction and retention. The company should also recognize and reward agents with high resolution rates to motivate them to continue providing excellent service.
</br>
&emsp;<li>Model selection: The company should use the Logistic Regression or Random Forest Classifier model for predicting churn as they have the highest accuracy scores and AUC-ROC scores. The Logistic Regression model has higher precision and recall scores, making it better at identifying churned customers. However, the specific needs of the problem and the costs associated with false positives and false negatives should be taken into account when selecting the appropriate model.
<br/>
<h3>Conclusions</h3>
<p>The analysis of two datasets showed that customer churn is a major problem for the company, with customers on month-to-month contracts, with shorter tenure, and without partners or dependents being more likely to churn. Senior citizens, customers with fiber optic internet, and those without online security or tech support are also at higher risk. The resolution rates for technical support and payment-related issues need improvement. Logistic Regression was evaluated as the best models for predicting churned customers.To reduce churn, the company should offer longer-term contracts, personalized promotions and incentives, senior discounts, improve internet services, and optimize payment methods. Improving the resolution rates for technical support and payment-related issues, and proper training of agents are crucial. Implementing these insights will lead to better business performance and growth.


-----
