from pandas import *
from types import *

data = read_csv('./adult_data.csv', header = 0)
data.columns = ["age", "workclass", "fnlwgt", "education",
                "education-num", "marital-status","occupation",
                "relationship", "race","sex","capital-gain",
                "capital-loss", "hours-per-week","native-country", "50k_threshold"]

workclass = ["Private","Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
             "Local-gov","State-gov", "Without-pay","Never-worked"]

education = ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school"
    , "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"]

marital_status = ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"]

sex = ["Male", "Female"]

occupation = ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
              "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct","Adm-clerical",
              "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"]

relationship = ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"]

race = ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]

native_country = ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)",
                  "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica",
                  "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti",
                  "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador",
                  "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"]

for x in range(len(sex)):
    data["sex"] = data["sex"].str.replace(sex[x], str(x))

data["sex"] = data["sex"].str.replace("?", "NaN")

for x in range(len(workclass)):
    data["workclass"] = data["workclass"].str.replace(workclass[x], str(x))

data["workclass"] = data["workclass"].str.replace("?", "NaN")

for x in range(len(education)):
    data["education"] = data["education"].str.replace(education[x], str(x))

data["education"] = data["education"].str.replace("?", "NaN")

for x in range(len(marital_status)):
    data["marital-status"] = data["marital-status"].str.replace(marital_status[x], str(x))


for x in range(len(occupation)):
    data["occupation"] = data["occupation"].str.replace(occupation[x], str(x))

data["occupation"] = data["occupation"].str.replace("?", "NaN")

for x in range(len(relationship)):
    data["relationship"] = data["relationship"].str.replace(relationship[x], str(x))

data["relationship"] = data["relationship"].str.replace("?", "NaN")

for x in range(len(race)):
    data["race"] = data["race"].str.replace(race[x], str(x))

data["race"] = data["race"].str.replace("?", "NaN")

for x in range(len(native_country)):
    data["native-country"] = data["native-country"].str.replace(native_country[x], str(x))

data["native-country"] = data["native-country"].str.replace("?", "NaN")

#https://stackoverflow.com/questions/25698710/replace-all-occurrences-of-a-string-in-a-pandas-dataframe-python

#print(data.columns[0])
#print(data.columns[1])
#print("native-country")
print(data)




