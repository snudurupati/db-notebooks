// Databricks notebook source
// MAGIC %md ## databricks for Data Engineers

// COMMAND ----------

// MAGIC %fs ls dbfs:/databricks-datasets/

// COMMAND ----------

val logFiles = "dbfs:/databricks-datasets/sample_logs/"	

// COMMAND ----------

import scala.util.matching.Regex

case class Row(ipAddress: String, clientIdentd: String, userId: String, dateTime: String, method: String, endpoint: String, protocol: String,responseCode: Int, contentSize: Long)
val pattern = """^(\S+) (\S+) (\S+) \[([\w:/]+\s[+\-]\d{4})\] "(\S+) (\S+) (\S+)" (\d{3}) (\d+)""".r

def parseLogLine(logline: String): Row = {
  val match1 = pattern.findAllIn(logline)
 if(!match1.isEmpty)
   Row(match1.group(1), match1.group(2), match1.group(3), match1.group(4), match1.group(5), match1.group(6), match1.group(7), match1.group(8).toInt, match1.group(9).toLong)
else Row(null, null, null, null, null, null, null, 0, 0)
}

// COMMAND ----------

val logFileRDD = sc.textFile(logFiles).cache()
logFileRDD.count
logFileRDD.partitions.size

// COMMAND ----------

val parsedRDD = logFileRDD.map(parseLogLine)
val logData = parsedRDD.toDF()
display(logData)

// COMMAND ----------

// MAGIC %run /Users/bill@databricks.com/credentials/redshift-scala

// COMMAND ----------

