// Databricks notebook source
// MAGIC %md ##**Step 1: Data Preparation**

// COMMAND ----------

// MAGIC %md ####Create **schemaRDDs** from the HIVE tables and register as **Spark temp tables** for further processing

// COMMAND ----------

val raw_data = sqlContext.table("german_credit_raw")
raw_data.registerTempTable("raw_data")

val credit_values = sqlContext.table("german_credit_value_modification")
credit_values.registerTempTable("new_values")

val checking_status = sqlContext.table("german_chk_acc_lkp")
checking_status.registerTempTable("checking_status")

// COMMAND ----------

// MAGIC %md ####map **credit history codes to values**, **checking account status codes to values** and create a joined SchemaRDD

// COMMAND ----------

/*val joinedRDD = sql("""select a3.description checking_acc_status,a1.duration_months, a2.description credit_history, a1.purpose, a1.credit_amount,
            a1.savings_acc_bond, a1.present_emp_since, a1.installment_rate, a1.personal_status, a1.other_debtors, a1.present_res_since, a1.property,
            a1.age, a1.other_inst_plans, a1.housing, a1.num_existing_credits, a1.job_type, a1.num_dependents, a1.telephone, a1.foreign_worker,
            (a1.credit_rating - 1) credit_rating
                        from raw_data a1 
                        join credit_values a2 
                        on a1.credit_history = a2.credit_history
                        join checking_status a3
                        on a1.checking_acc_status = a3.checking_acc_status""")*/
val joinedRDD = sql("""select a2.description checking_acc_status, a1.duration_months, a3.NewValue credit_history, a4.NewValue purpose, a1.credit_amount,
a5.NewValue savings_acc_bond, a14.NewValue present_emp_since, a1.installment_rate, a6.NewValue personal_status, a7.NewValue other_debtors, a1.present_res_since, a8.NewValue property, a1.age, a9.NewValue other_inst_plans, a10.NewValue housing, a1.num_existing_credits, a11.NewValue job_type, a1.num_dependents, a12.NewValue telephone, a13.NewValue foreign_worker,(a1.credit_rating - 1) credit_rating
from raw_data a1 
join checking_status a2 
on a1.checking_acc_status = a2.checking_acc_status
join new_values a3
on a1.credit_history = a3.OldValue 
join new_values a4
on a1.purpose = a4.OldValue 
join new_values a5
on a1.savings_acc_bond = a5.OldValue 
join new_values a6
on a1.personal_status = a6.OldValue 
join new_values a7
on a1.other_debtors = a7.OldValue 
join new_values a8
on a1.property = a8.OldValue 
join new_values a9
on a1.other_inst_plans = a9.OldValue 
join new_values a10
on a1.housing = a10.OldValue 
join new_values a11
on a1.job_type = a11.OldValue 
join new_values a12
on a1.telephone = a12.OldValue 
join new_values a13
on a1.foreign_worker = a13.OldValue 
join new_values a14
on a1.present_emp_since = a14.OldValue""")

// COMMAND ----------

joinedRDD.count()
joinedRDD.registerTempTable("joined_data")


// COMMAND ----------

// MAGIC %sql select * from joined_data limit 10

// COMMAND ----------

// MAGIC %md #### Import the appropriate MLlib packages

// COMMAND ----------

import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors

// COMMAND ----------

// MAGIC %md #### Create a LebeledPoint RDD with Label followed by feature vector

// COMMAND ----------

//val l_iris = iris.map(x => LabeledPoint(label_map(x._1), Vectors.dense(x._2)))
//joinedRDD.map(s => s(20).asInstanceOf[Int].toDouble).take(5)
val labeledPointRDD = joinedRDD.map(s => LabeledPoint(s(20).asInstanceOf[Int].toDouble, Vectors.dense(s(1).asInstanceOf[Int].toDouble, s(4).asInstanceOf[Int].toDouble, s(12).asInstanceOf[Int].toDouble, s(15).asInstanceOf[Int].toDouble)))
labeledPointRDD.take(5)

// COMMAND ----------

// MAGIC %md #### Randomly split data into 70/30 **training** and **test** datasets

// COMMAND ----------

val splits = labeledPointRDD.randomSplit(Array(0.7, 0.3))
val (trainingData, testData) = (splits(0), splits(1))
println("rows in training and test data respectively %d, %d".format(trainingData.count, testData.count))
trainingData.cache

// COMMAND ----------

// MAGIC %md ####**Configure the decision tree and train the model**
// MAGIC #####The descision tree training has two parts, **Category** defined below and **input** which in this cases needs to be a **RDD[LabeledPoint]**
// MAGIC LabeledPoint = (Label, Feature vector) in LIBSVM format
// MAGIC ######**algorithm** = Classification
// MAGIC ######**impurity** = Gini
// MAGIC ######**maxDepth** = 4
// MAGIC ######**numClassessForClassification** = 2 (binary classification)

// COMMAND ----------

val numClasses = 2
val categoricalFeaturesInfo = Map[Int, Int]()
val impurity = "gini"
val maxDepth = 4
val maxBins = 60

val model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)


// COMMAND ----------

// MAGIC %md ####test the model using the test data

// COMMAND ----------

val labelAndPreds = testData.map{ p => 
                                val prediction = model.predict(p.features)
                                (p.label, prediction) }
val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble/testData.count()

println("Test error is %s\n".format(testErr))
println("Learned classification tree model:\n%s".format(model.toDebugString))

// COMMAND ----------

labelAndPreds.take(10)