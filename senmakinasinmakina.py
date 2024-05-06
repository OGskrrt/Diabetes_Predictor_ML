import warnings

# verileri düzenlemek için
import numpy as np
import pandas as pd

# verileri görselleştirmek için
import seaborn as sns
import matplotlib.pyplot as plt

#Calssifier'lar verileri sınıflandırmak için
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# verileri ölçeklendirmek ve düzenlemek için
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder

# başarı oranı için
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV

warnings.simplefilter(action='ignore', category=Warning)


# veri output düzenlemek için ayar
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

############# exploratory data analysis EDA  #################

# csv dosyasını okuduk
df = pd.read_csv("diabetes.csv")
"""
# ilk kısımları görmek için
print(df.head())
# son kısımları göstermek için son 10 kısım default 5
print(df.tail(10))
# satır sütün saysı
print(df.shape)
"""
# Outcome kolonunun satır sayısını verdi
#print(df["Outcome"].value_counts())
# aynısın görselleştirilmiş hali
#sns.countplot(x="Outcome", data=df)
#plt.show()
# aynısını yüzdelik dilimini görmek için
#print(100*df["Outcome"].value_counts()/len(df))
# ortlama standart satma cart curt gösteren fonksiyon, .t tranpoze almak için, daha iyi duruyo
#print(df.describe().T)


# outocome hariç tüm kolonları almak için
def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show(block=True)


cols = [col for col in df.columns if "Outcome" not in col]

for col in cols:
    plot_numerical_col(df, col)



# target kolononua göre diğer kolonların ortalaması
#def target_summary_with_num(dataframe, target, numerical_col):
    #print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


#for col in cols:
#target_summary_with_num(df, "Outcome", col)

# boş kısımların toplamı
#print(df.isnull().sum())

############# exploratory data analysis EDA sonu #################


############ data preprocessing ##############
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    # ooutlier_thresholds ile çeyreklilk değerler belirlenip alt ve üst limitler belirleniyor,
    # limitlerin arasında olmayan değer var ise aykırı değer oluyor
    # .any kısmı tüm satırlara bakıyor
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# check outlier ile aykırı değer olup olmadığına bak
#for col in cols:
#    print(col, check_outlier(df, col))

# aykırı değeri üst limit veya alt limit ile değiştir
replace_with_thresholds(df, "Insulin")

#for col in cols:
#   print(col, check_outlier(df, col))

# veriyi ölçeklendirmek için
for col in cols:
    df[col] = RobustScaler().fit_transform(df[[col]])

############ data preprocessing ##############

############### model decision ###############
#sadece outcome kolonu
y = df["Outcome"]
# kalan tüm kolonlar
X = df.drop(["Outcome"], axis=1)

# veriyi 80-20 olarak böldük, 80 ile eğitilecek 20 ile test edilecek
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)


# tüm modelleri aldık
cart_model = DecisionTreeClassifier(random_state=42).fit(X, y)
catboost_model = CatBoostClassifier(random_state=42, verbose=False)
gbm_model = GradientBoostingClassifier(random_state=42)
lgbm_model = LGBMClassifier(random_state=42)
rf_model = RandomForestClassifier(random_state=42)
xgboost_model = XGBClassifier(random_state=42, use_label_encoder=False)


# cross validation yöntemi ile her bir modelin accuracy'sini ölçtük
cv_results = cross_validate(cart_model, X, y, cv=10, scoring=["accuracy"])
print("Cart-Model accuracy score: ", cv_results['test_accuracy'].mean())

cv_results = cross_validate(catboost_model, X, y, cv=10, scoring=["accuracy"])
print("Catboost-Model accuracy score: ", cv_results['test_accuracy'].mean())

cv_results = cross_validate(gbm_model, X, y, cv=10, scoring=["accuracy"])
print("GBM-Model accuracy score: ", cv_results['test_accuracy'].mean())

cv_results = cross_validate(lgbm_model, X, y, cv=10, scoring=["accuracy"])
print("LightGBM-Model accuracy score: ", cv_results['test_accuracy'].mean())

cv_results = cross_validate(rf_model, X, y, cv=10, scoring=["accuracy"])
print("RF-Model accuracy score: ", cv_results['test_accuracy'].mean())

cv_results = cross_validate(xgboost_model, X, y, cv=10, scoring=["accuracy"])
print("XGBoost-Model accuracy score: ", cv_results['test_accuracy'].mean())

############ sonuçlara göre catbosst en iyis, hiperoptimizasyon kısmına giriyoruz

############# hiper parametre oprimizastonu############


catboost_params = {"iterations": [100, 200, 500],
                   "learning_rate": [0.01, 0.1, 0.001],
                   "depth": [2, 3, 6, 8]}

# model ismi ve model parametresini ver, tüm kombinasyonlar için ayrı ayrı corss-validation yap, sonucu grid haline dönüştür
catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=10, n_jobs=-1, verbose=True).fit(X, y)

# en iyi parametreyi modelin içine göm ve çalıştır
catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=42).fit(X, y)


# sonuca bak
cv_results = cross_validate(catboost_final, X, y, cv=10, scoring=["accuracy"])
print("===== Hiperparametre Optimizasyonu Sonrası =====")
print("accuracy score: ", cv_results["test_accuracy"].mean())

# ilişkiyi görselleştirme, korelasyon
correlation_matrix = df.corr()

plt.figure(figsize=(20, 16))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, annot_kws={'size': 10}),
plt.title('Attribute Correlation Map')
plt.show()