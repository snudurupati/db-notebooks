// Databricks notebook source
// MAGIC %fs ls /databricks-datasets/Rdatasets/data-001/datasets.csv

// COMMAND ----------

val dataPath = "/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv"
val diamonds = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(dataPath)

// COMMAND ----------

diamonds.printSchema

// COMMAND ----------

display(diamonds)

// COMMAND ----------

val df1 = diamonds.groupBy("cut", "color").avg("price")
val df2 = df1.join(diamonds, "color").select("avg(price)", "carat")

// COMMAND ----------

df2.show()

// COMMAND ----------

df2.explain

// COMMAND ----------

df2.count()

// COMMAND ----------

