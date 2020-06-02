import sys

from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator,MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression, NaiveBayes, RandomForestClassifier, LinearSVC, MultilayerPerceptronClassifier
from pyspark.ml.regression import LinearRegression


def cargar_datos(sc, sqlc, path):
    headers = sc.textFile("/user/datasets/ecbdl14/ECBDL14_IR2.header").filter(lambda line: "@inputs" in line in line)
    headers = headers.flatMap(lambda line: line.replace(",", "").split())
    headers= headers.collect()
    del headers[0]
    headers.append("class")

    df = sqlc.read.csv('/user/datasets/ecbdl14/ECBDL14_IR2.data', header=False, inferSchema=True)

    for i, colname in enumerate(df.columns):
        df = df.withColumnRenamed(colname, headers[i])

    df = df.select(
        "PSSM_r2_3_E", "PSSM_r1_4_L", "PSSM_r1_-4_I",
        "PSSM_r2_-4_Y", "PSSM_r1_-1_F", "PSSM_r1_1_S", "class")

    df.write.csv(path, header=True)

    return df

def preprocesado(df):

  vectorAssembler = VectorAssembler(inputCols=[
      "PSSM_r2_3_E", "PSSM_r1_4_L", "PSSM_r1_-4_I",
      "PSSM_r2_-4_Y", "PSSM_r1_-1_F", "PSSM_r1_1_S"
  ], outputCol="features")
  #preprocessed_df = vectorAssembler.transform(df).select("features", "class")
  preprocessed_df = vectorAssembler.transform(df).selectExpr('features as features','class as label').select("features", "label")

  return preprocessed_df

def undersampling(df):

  negativoDf = df.filter(df['label'] == 0)
  positivoDf = df.filter(df['label'] == 1)
  sampleRatio = float(positivoDf.count()) / float(df.count())
  negativoSampleDf = negativoDf.sample(False, sampleRatio)
  dataframe = positivoDf.union(negativoSampleDf)

  return dataframe

def naive_bayes(train, test, smoothing, modelType):

  nb = NaiveBayes(smoothing=smoothing, modelType=modelType)

  # Entrenamos el modelo
  model = nb.fit(train)

  predictions = model.transform(test)
  # evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
  evaluator = BinaryClassificationEvaluator()
  accuracy = evaluator.evaluate(predictions)

  return accuracy

def random_forest(train, test, numTrees, impurity):
  # Entrenamos el modelo.
  rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=numTrees, impurity=impurity, seed=13)
  model = rf.fit(train)
  evaluator = BinaryClassificationEvaluator()
  accuracy = evaluator.evaluate(model.transform(test))

  return accuracy

def perceptron_multicapa(train, test, capas, num_iter, tamlot):

  layers = capas

  trainer = MultilayerPerceptronClassifier(
        maxIter=num_iter, layers=layers, blockSize=tamlot, seed=13)
  # Entrenamos el modelo
  model = trainer.fit(train)
  # compute accuracy on the test set
  result = model.transform(test)
  predictionAndLabels = result.select('prediction', 'label')
  evaluator = MulticlassClassificationEvaluator(metricName='accuracy')
  accuracy = evaluator.evaluate(predictionAndLabels)

  return accuracy


def linear_svc(train, test, maxIter, regParam):

  lsvc = LinearSVC(maxIter=maxIter, regParam=regParam)
  # Fit the model
  lsvcModel = lsvc.fit(train)

  return lsvcModel

def print_result(accuracy, classification):
  print("Test Error "+ classification +" = %g" % (1.0 - accuracy))
  print("Test accuracy "+ classification +" :",  accuracy)

if __name__ == "__main__":

  # create Spark context with Spark configuration
  conf = SparkConf().setAppName("Practica 4 - Cristobal Rodriguez")
  sc = SparkContext(conf=conf)
  sqlc = SQLContext(sc)

  # Comprobamos si existe el fichero sino lo cargamos
 fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())
  path = './filteredC.small.training'
  if fs.exists(sc._jvm.org.apache.hadoop.fs.Path(path)):
    print("Existe el fichero. Se procede a su lectura")
    df = sqlc.read.csv(path, header=True, inferSchema=True)
  else:
    df = cargar_datos(sc, sqlc, path)

  df.show()

  preproDf = preprocesado(df)

  # Balanceado por Random Undersample (RUS)
  balanceadoDf = undersampling(preproDf)

  train, test = balanceadoDf.randomSplit([0.7, 0.3], seed = 13)

  results = {}

  #acc_nb =naive_bayes(train, test, 1.0, 'multinomial')
  #print_result(acc_nb, "naive_bayes 1.0 multinomial")
  #results.update({'naive_bayes 1.0 multinomial':acc_nb})
  #acc_nb =naive_bayes(train, test, 1.0, 'bernoulli')
  #results.update({'naive_bayes 1.0 bernoulli':acc_nb})
  #acc_nb =naive_bayes(train, test, 3.0, 'multinomial')
  #results.update({'naive_bayes 3.0 multinomial':acc_nb})
  #acc_nb =naive_bayes(train, test, 3.0, 'bernoulli')
  #results.update({'naive_bayes 3.0 bernoulli':acc_nb})

  acc_rf= random_forest(train, test, numTrees=30, impurity='entropy')
  results.update({'random_forest 30 entropy':acc_rf})
  acc_rf= random_forest(train, test, numTrees=100, impurity='entropy')
  results.update({'random_forest 100 entropy':acc_rf})
  acc_rf= random_forest(train, test, numTrees=30, impurity='gini')
  results.update({'random_forest 30 gini':acc_rf})
  acc_rf= random_forest(train, test, numTrees=100, impurity='gini')
  results.update({'random_forest 100 gini':acc_rf})
  # print_result(acc_rf, "random_forest 100 gini")

  acc_pm  = perceptron_multicapa(balanceado_train, test, [6, 20, 30 , 2], 50, 64)
  results.update({'perceptron_multicapa, capas = [6, 20, 30 , 2], numiter=50, tamlot=64': acc_pm})
  acc_pm  = perceptron_multicapa(balanceado_train, test, [6, 20, 2], 50, 64)
  results.update({'perceptron_multicapa, capas = [6, 20, 2], numiter=100, tamlot=128 ': acc_pm})
  acc_pm  = perceptron_multicapa(balanceado_train, test, [6, 50, 30, 2], 50, 64)
  results.update({'perceptron_multicapa, capas = [6, 50, 30, 2], numiter=50, tamlot=64': acc_pm})
  acc_pm  = perceptron_multicapa(balanceado_train, test, [6, 50, 2], 100, 128)
  results.update({'perceptron_multicapa, capas = [6, 50, 2], numiter=100, tamlot=128 ': acc_pm})



  lsvcModel = linear_svc(train, test, 10, 1.0) 
  lsvcModel2 = linear_svc(train, test, 20, 0.05)
  lsvcModel3 = linear_svc(train, test, 20, 1.0) 
  lsvcModel4 = linear_svc(train, test, 10, 0.05)  

  # Mostra los resultados obtenidos de los algorimos de clasificacion

  print("Coefficients: " + str(lsvcModel.coefficients))
  print("Intercept: " + str(lsvcModel.intercept))
  print("Coefficients: " + str(lsvcModel2.coefficients))
  print("Intercept: " + str(lsvcModel2.intercept))
  print("Coefficients: " + str(lsvcModel3.coefficients))
  print("Intercept: " + str(lsvcModel3.intercept))
  print("Coefficients: " + str(lsvcModel4.coefficients))
  print("Intercept: " + str(lsvcModel4.intercept))
  
  for key,acc_rf in results.items():
     print_result(acc_rf, key)

  sc.stop()
