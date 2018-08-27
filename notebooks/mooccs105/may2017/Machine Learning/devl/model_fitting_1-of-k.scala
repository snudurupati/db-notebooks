// Databricks notebook source
// MAGIC %md #### Applying a learning algorithm and fitting a model
// MAGIC ##### Since this is a **binomial classification** problem, we will be using a **classification algorithm** that uses **decision trees**

// COMMAND ----------

// MAGIC %run "/snudurupati/Machine Learning/feature_extraction"

// COMMAND ----------

// MAGIC %md #### Split data randomly into 70/30 **training** and **test** datasets

// COMMAND ----------

val splits = dataLP.randomSplit(Array(0.7, 0.3))
val (trainingData, testData) = (splits(0), splits(1))
trainingData.cache()
println("rows in training and test data respectively %d, %d".format(trainingData.count, testData.count))

// COMMAND ----------

// MAGIC %md ####**Configure the decision tree and train the model**
// MAGIC #####The descision tree training has two parts, **Category** defined below and **input** which needs to be a **LabeledPoint RDD**
// MAGIC LabeledPoint = (Label, Feature vector)
// MAGIC ######**algorithm** = Classification
// MAGIC ######**impurity** = Gini
// MAGIC ######**maxDepth** = 4
// MAGIC ######**numClassessForClassification** = 2 (binary classification)

// COMMAND ----------

import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel

val numClasses = 2
val categoricalFeaturesInfo = Map[Int, Int](0 -> 4)
val impurity = "gini"
val maxDepth = 4
val maxBins = 32

val model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)


// COMMAND ----------

// MAGIC %md #### Test the model using the test data

// COMMAND ----------

val labelAndPreds = testData.map{ p => 
                                val prediction = model.predict(p.features)
                                (p.label, prediction) }
val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble/testData.count()

println("Test error is %s\n".format(testErr))
println(dataLP.first())
println("Learned classification tree model:\n%s".format(model.toDebugString))

// COMMAND ----------

model.topNode