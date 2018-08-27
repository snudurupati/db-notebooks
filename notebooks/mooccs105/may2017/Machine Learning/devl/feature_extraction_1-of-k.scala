// Databricks notebook source
// MAGIC %md ##Feature extraction from categories using **1-of-k** encoding

// COMMAND ----------

// MAGIC %md #####Import from results of the data munging step

// COMMAND ----------

// MAGIC %run "/snudurupati/Machine Learning/1. data_preparation"

// COMMAND ----------

// MAGIC %md ####Extract 1 categorical feature using 1-of-k encoding and other numerical features from prepared data.

// COMMAND ----------

val checkingStatus = prepared_data.map(r => r(0)).distinct.collect.zipWithIndex.toMap
val numCheckingStatus = checkingStatus.size
checkingStatus.foreach(println)


// COMMAND ----------

// MAGIC %md #### Create RDD of LebeledPoint combining both categorical and numerical features.

// COMMAND ----------

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

val dataLP = prepared_data.map{r => 
  val label = r(r.size-1).asInstanceOf[Int]
  val chkStatusIdx = checkingStatus(r(0))
  val chkStatusFeatures = Array.ofDim[Double](numCheckingStatus)
  chkStatusFeatures(chkStatusIdx) = 1.0
  val otherFeatures = r.slice(12, r.size - 1).map(d => d.asInstanceOf[Int].toDouble)
  val features = chkStatusFeatures ++ otherFeatures
  LabeledPoint(label, Vectors.dense(features))
}
/*prepared_data.map(r => r.slice(12, r.size-1).map(d => d.asInstanceOf[Int].toDouble)).take(5).foreach(println)
dataLP.take(10).foreach(println)*/