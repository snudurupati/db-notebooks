// Databricks notebook source
// MAGIC %md ##Feature extraction from text using term frequency (TF)

// COMMAND ----------

// MAGIC %md #####Import from results of the data munging step

// COMMAND ----------

// MAGIC %run "/snudurupati/Machine Learning/1. data_preparation"

// COMMAND ----------

// MAGIC %md ####Extract 20 features from each of good and bad credit RDDs.

// COMMAND ----------

import org.apache.spark.mllib.feature.HashingTF

val tf = new HashingTF(100)
val gcfeatureVec = good_credit.map(r => tf.transform(r))
val bcfeatureVec = bad_credit.map(r => tf.transform(r))

gcfeatureVec.take(10).foreach(println)
bcfeatureVec.take(10).foreach(println)

// COMMAND ----------

// MAGIC %md #### Create RDDs of LebeledPoint for good and bad credit features separately and combine them to give training data

// COMMAND ----------

import org.apache.spark.mllib.regression.LabeledPoint

val good_creditLP = gcfeatureVec.map(features => LabeledPoint(1, features))
val bad_creditLP = bcfeatureVec.map(features => LabeledPoint(0, features))
val featureVector = good_creditLP ++ bad_creditLP

//featureVector.take(10).foreach(println)
