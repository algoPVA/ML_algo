import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('titanic.csv')


df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'],axis = 1, inplace = True)

# df['Embarked'].fillna('S', inplace = True)
# df.fillna({'Embarked':'S'}, inplace = True)
df['Embarked'] = df['Embarked'].fillna('S')

age_1 = df[df['Pclass'] == 1]['Age'].median()
age_2 = df[df['Pclass'] == 2]['Age'].median()
age_3 = df[df['Pclass'] == 3]['Age'].median()

def fill_age(row):
   if pd.isnull(row['Age']):
       if row['Pclass'] == 1:
           return age_1
       if row['Pclass'] == 2:
           return age_2
       return age_3
   return row['Age']


df['Age'] = df.apply(fill_age, axis = 1)

def fill_sex(sex):
    if sex == 'male':
        return 1
    return 0


df['Sex'] = df['Sex'].apply(fill_sex)

df[list(pd.get_dummies(df['Embarked']).columns)] = pd.get_dummies(df['Embarked'])

df.drop('Embarked', axis = 1, inplace = True)

def is_alone(row):
    if row['SibSp'] + row['Parch'] == 0:
        return 1
    return 0

df['Alone'] = df.apply(is_alone, axis = 1)

x = df.drop('Survived', axis = 1) # Данные о пассажирах
y = df['Survived']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)