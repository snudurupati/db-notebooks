// Databricks notebook source
// MAGIC %md ##Feature extraction from text using term frequency (TF)

// COMMAND ----------

val sentences = List("Software upgrades and outdated applications that don't work on new platforms are just a fact of life for people who use computers and other devices.", "DARPA, however, wants to change that by making software systems that can run for over a century without getting updates from their developers and despite upgrades in hardware.", "Pentagon's mad science department has recently announced that it has begun a four-year research to figure out what algorithms are necessary to create software that can dynamically adapt to changes.")
val linesRDD = sc.parallelize(sentences)

// COMMAND ----------

import org.apache.spark.mllib.feature.HashingTF

val tf = new HashingTF(10)
val featureVec = linesRDD.map(line => tf.transform(line.split(" ")))
featureVec.collect.foreach(println)