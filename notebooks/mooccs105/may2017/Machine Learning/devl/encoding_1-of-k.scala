// Databricks notebook source
// MAGIC %md #### Extracting categorical features using **1-of-k** encoding

// COMMAND ----------

val occupations = List("Secretaries and administrative assistants", "Registered nurses", "Elementary and middle school teachers", "Cashiers", 
"Nursing, psychiatric, and home health aides", "Retail salespersons", "First-line supervisors/managers of retail sales workers", "Waiters and Waitresses","Maids and housekeeping cleaners", "Customer service representatives", "Child-care workers", "Bookkeeping, accounting, and auditing clerks", "Receptionists and Information clerks","First-line supervisors/managers of office and administrative support", "Managers, all others", "Accountants and auditors", "Teacher assistants", "Cooks", "Office clerks, general", "Personal and home care aides")
val records = sc.parallelize(occupations)



// COMMAND ----------

val categories = occupations.zipWithIndex.toMap
println(categories.size)
categories.foreach(println)

// COMMAND ----------

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint

val featureVec = records.map{r => 
  val label = if (r.size - 1 > 9) 1 else 0 
  val categoryIdx = categories(r)
  val categoryFeatures = Array.ofDim[Double](categories.size)
  categoryFeatures(categoryIdx) = 1.0
  LabeledPoint(label, Vectors.dense(categoryFeatures))
}
featureVec.take(5).foreach(println)
