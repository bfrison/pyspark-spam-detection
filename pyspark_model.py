import matplotlib.pyplot as plt
import pyspark.sql.functions as F
import seaborn as sns
from pyspark import SparkContext
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import Tokenizer, Word2Vec
from pyspark.sql.session import SparkSession
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)

sc = SparkContext("local", "pyspark demo")
sc.setLogLevel('WARN')
spark = SparkSession(sc)

df = (
    spark.read.options(header=True, inferSchema=True)
    .csv('spam.csv')
    .drop('_c2', '_c3', '_c4')
    .toDF('label', 'text')
    .na.drop()
)
df = df.withColumn('label', F.when(df.label == 'spam', 1).when(df.label == 'ham', 0))

t = Tokenizer(inputCol='text', outputCol='words')
df = t.transform(df)

word2Vec = Word2Vec(inputCol='words', outputCol='vectors', vectorSize=8, seed=42)
word2Vec_model = word2Vec.fit(df)
df = word2Vec_model.transform(df)

train, test = df.randomSplit([0.8, 0.2], 42)

rf = RandomForestClassifier(featuresCol='vectors', labelCol='label', seed=42)
rf_model = rf.fit(train.na.drop())

test_pred = rf_model.transform(test)

t.write().overwrite().save('models/tokenizer')
word2Vec_model.write().overwrite().save('models/word2Vec_model')
rf_model.write().overwrite().save('models/rf_model')
train.toPandas().to_csv('results/train.csv')
test_pred.toPandas().to_csv('results/test.csv')


def c_report(y_true, y_pred):
    print("Classification Report")
    print(classification_report(y_true, y_pred))
    acc_sc = accuracy_score(y_true, y_pred)
    print("Accuracy : " + str(acc_sc))
    print(f'AUC score : {roc_auc_score(y_true, y_pred):f}')


def plot_confusion_matrix(y_true, y_pred):
    mtx = confusion_matrix(y_true, y_pred)
    sns.heatmap(mtx, annot=True, fmt='d', linewidths=0.5, cmap="Blues", cbar=False)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('results/confusion_matrix.png')


c_report(test_pred.select('label').collect(), test_pred.select('prediction').collect())

plot_confusion_matrix(
    test_pred.select('label').collect(), test_pred.select('prediction').collect()
)
