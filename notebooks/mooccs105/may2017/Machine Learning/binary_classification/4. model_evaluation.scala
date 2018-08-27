// Databricks notebook source
// MAGIC %md ###Evaluating the performance of the model

// COMMAND ----------

// MAGIC %run "/snudurupati/Machine Learning/3. model_fitting"

// COMMAND ----------

// MAGIC %md ####Accuracy and Prediction Error

// COMMAND ----------

val labelAndPreds = testData.map{ p => 
                                val prediction = model.predict(p.features)
                                (p.label, prediction) }
val dtAccuracy = labelAndPreds.filter(r => r._1 == r._2).count.toDouble/testData.count()
//val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble/testData.count()

println("The accuracy of the model is %2.2f%%\n".format(dtAccuracy*100))

// COMMAND ----------

// MAGIC %md ####Precision and Recall
// MAGIC ##### In binary classification Precision = **(num. true positives)/(num true positives + num false positives)**, thus precision = 1.0 when there are no false positives
// MAGIC #####Recall = **(num. true positives)/(num true positives + num false negatives)**, thus recall = 1.0 when there are no false negatives

// COMMAND ----------

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

val metrics = Seq(model).map{ model =>
  val labelsAndpreds = testData.map { point =>
  val score = model.predict(point.features)
    (if (score > 0.5) 1.0 else 0.0, point.label)
  }
  val metrics = new BinaryClassificationMetrics(labelsAndpreds)
  (model.getClass.getSimpleName, metrics.areaUnderPR,metrics.areaUnderROC)
}

// COMMAND ----------

metrics.foreach { case (m, pr, roc) =>
  println("Area under PR: %2.4f%%\nArea under ROC: %2.4f%%".format(pr * 100.0, roc * 100.0))
  //println(f"$m, Area under PR: ${pr * 100.0}%2.4f%%, Area under ROC: ${roc * 100.0}%2.4f%%")
}