// Databricks notebook source
val iris_csv = sc.textFile("/databricks-datasets/Rdatasets/data-001/csv/datasets/iris.csv")

// COMMAND ----------

val iris_rows = sqlContext.sql(s"""
SELECT 
  species, sepal_length, sepal_width, petal_length, petal_width
FROM iris 
WHERE sepal_length IS NOT NULL
""")

// COMMAND ----------

iris_rows.take(3)

// COMMAND ----------

import org.apache.spark.rdd.RDD
val iris: RDD[(String, Array[Double])] = iris_rows.map { s =>
  (s(0).asInstanceOf[String].stripPrefix("\"").stripSuffix("\""), s.slice(1, 5).asInstanceOf[List[Double]].toArray)
}

iris.take(2)

// COMMAND ----------

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

// COMMAND ----------

val label_map = Map("setosa" -> 1.0, "versicolor" -> 2.0, "virginica" -> 0.0)
val l_iris = iris.map(x => LabeledPoint(label_map(x._1), Vectors.dense(x._2)))
iris.take(5)

// COMMAND ----------


l_iris.take(5)