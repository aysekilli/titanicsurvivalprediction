import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")

# Genel bakış
print(df.shape)
print(df.head())
print(df.info())
print(df.describe())

# Eksik değerler
print(df.isnull().sum())

# Hayatta kalma oranı
print(df['Survived'].value_counts())

# Cinsiyet vs hayatta kalma
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Survival Rate by Gender')
plt.savefig('survival_gender.png')
plt.clf()

# Pclass vs hayatta kalma
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival Rate by Class')
plt.savefig('survival_class.png')
plt.clf()

# Yaş dağılımı
sns.histplot(df['Age'].dropna(), bins=30, kde=True)
plt.title('Age Distribution')
plt.savefig('age_dist.png')
plt.clf()

# Yaş vs hayatta kalma
sns.boxplot(x='Survived', y='Age', data=df)
plt.title('Age vs Survival')
plt.savefig('age_survival.png')
plt.clf()

# Korelasyon matrisi
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation.png')
plt.clf()

print("Bitti!")