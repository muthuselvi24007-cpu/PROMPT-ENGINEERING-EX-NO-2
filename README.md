# EX-02-Cross-Platform-Prompting-Evaluating-Diverse-Techniques-in-AI-Powered-Text-Summarization

## AIM
To evaluate and compare the effectiveness of prompting techniques (zero-shot, few-shot, chain-of-thought, role-based) across different AI platforms (e.g., ChatGPT, Gemini, Claude, Copilot) in a specific task: text summarization.

## SCENARIO:
You are part of a content curation team for an educational platform that delivers quick summaries of research papers to undergraduate students. Your task is to summarize a 500-word technical article on "The Basics of Blockchain Technology" using multiple AI platforms and prompting strategies.

Your goal is to determine which combination of prompting technique + platform provides the best summary in terms of:

Accuracy

Coherence

Simplicity

Speed

User experience

## OUTPUT
import pandas as pd df=pd.read_csv(r"C:\Users\Downloads\titanic_dataset.csv") df
<img width="1310" height="504" alt="image" src="https://github.com/user-attachments/assets/e17f1847-efb8-4823-bf9e-9bef45c9f81d" />
df.shape image df.set_index("PassengerId",inplace=True) df
<img width="1351" height="542" alt="image" src="https://github.com/user-attachments/assets/6184a00e-fd94-4830-9b3c-99990d27de84" />
df.nunique
<img width="1115" height="829" alt="image" src="https://github.com/user-attachments/assets/04a8951b-ba3e-4052-aacb-b6ed39a32e61" />
df['Sex'].value_counts()
<img width="1225" height="87" alt="image" src="https://github.com/user-attachments/assets/6fd79747-1eb3-4ddd-b1d8-2ab9f3bc47b1" />
df.Survived.unique()
<img width="1209" height="43" alt="image" src="https://github.com/user-attachments/assets/e2fa3044-8025-4b56-866c-71b129f95c50" />
df.rename(columns={"Sex":"Gender"},inplace=True)

df
<img width="1326" height="515" alt="image" src="https://github.com/user-attachments/assets/a598d830-5601-409e-8e3b-c07f5168d511" />
import seaborn as sns
sns.countplot(data=df)
<img width="1108" height="588" alt="image" src="https://github.com/user-attachments/assets/9abc9b19-8bf5-45e4-82b2-40064a1c4d3f" />
sns.countplot(x="Survived",hue="Gender",data=df)
<img width="1110" height="572" alt="image" src="https://github.com/user-attachments/assets/9f2b22c7-cb61-44ae-b44f-ee27f824e651" />
sns.catplot(x="Survived",hue="Gender",data=df,kind="count")
<img width="1131" height="643" alt="image" src="https://github.com/user-attachments/assets/05ba9fbb-9958-443a-83c5-bad11f47d94e" />
sns.catplot(x="Survived",hue="Gender",data=df,kind="violin")
<img width="1132" height="637" alt="image" src="https://github.com/user-attachments/assets/704e6198-5652-4a12-98d8-4d8d33266812" />
sns.boxplot(data=df)
<img width="1264" height="549" alt="image" src="https://github.com/user-attachments/assets/ac8b5d25-c966-44bd-9d5e-6ecd52909cef" />
df.boxplot(column="Survived",by="Gender")
<img width="1275" height="594" alt="image" src="https://github.com/user-attachments/assets/7b296cdb-8a04-46ef-a767-9ec3e7083397" />
sns.scatterplot(data=df)
<img width="962" height="545" alt="image" src="https://github.com/user-attachments/assets/71acb9c4-ddb0-49b6-8c50-bbb60ecfb513" />
sns.scatterplot(x=df['Age'],y=df['Fare'])
<img width="1267" height="552" alt="image" src="https://github.com/user-attachments/assets/fb1b3a33-ba16-4973-b13e-923040de1f2e" />
sns.jointplot(x='Age',y='Fare',data=df)
<img width="1256" height="763" alt="image" src="https://github.com/user-attachments/assets/e3eeeade-3cd7-432d-ad2a-34c98cef43cb" />
sns.jointplot(x='Age',y='Fare',data=df,kind="kde")
<img width="1279" height="762" alt="image" src="https://github.com/user-attachments/assets/1eed7aad-2292-4121-8e40-2876e8f2b22b" />
sns.jointplot(x='Age',y='Fare',data=df,kind="hist")
<img width="1030" height="759" alt="image" src="https://github.com/user-attachments/assets/1537232a-6932-41d5-be19-d06c7f32c977" />
sns.catplot(x='Gender',col='Survived',data=df,kind='count',color='green')
<img width="1298" height="644" alt="image" src="https://github.com/user-attachments/assets/ffc0d46a-6a34-41a1-b5be-c5249931c262" />
sns.pairplot(data=df)
<img width="901" height="927" alt="image" src="https://github.com/user-attachments/assets/87473aa1-e04e-462b-bc45-4a48fe72e694" />
corr1=df.select_dtypes(include=["number"]).corr()
sns.heatmap(corr1,annot=True)
<img width="1066" height="632" alt="image" src="https://github.com/user-attachments/assets/9a11a4d8-f21d-4bfb-9a83-fdbc9ad45ee1" />
sns.catplot(x='Gender',col='Survived',data=df,kind='count',hue="Pclass")
<img width="1353" height="640" alt="image" src="https://github.com/user-attachments/assets/2ab9a844-1eaf-459a-9362-8f3ea04ea415" />

import matplotlib.pyplot as plt
fig,ax1=plt.subplots(figsize=(8,5))

pt=sns.boxplot(ax=ax1,x='Pclass',y='Age',hue='Gender',data=df)
<img width="1159" height="558" alt="image" src="https://github.com/user-attachments/assets/97d90a95-cc5f-4a13-a10e-7586c5ccfb43" />






## RESULT
Thus performing Exploratory Data Analysis on the given data set.


