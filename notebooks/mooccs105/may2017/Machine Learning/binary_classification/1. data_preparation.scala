// Databricks notebook source
val raw_data = sqlContext.table("german_credit_raw")
raw_data.registerTempTable("raw_data")

val credit_values = sqlContext.table("german_credit_value_modification")
credit_values.registerTempTable("new_values")

val checking_status = sqlContext.table("german_chk_acc_lkp")
checking_status.registerTempTable("checking_status")


// COMMAND ----------

// MAGIC %md ####map **credit history codes to values**, **checking account status codes to values** and create a joined SchemaRDD

// COMMAND ----------

val prepared_data = sql("""select a2.description checking_acc_status, a3.NewValue credit_history, a4.NewValue purpose, a5.NewValue savings_acc_bond, a14.NewValue present_emp_since, a6.NewValue personal_status, a8.NewValue property, a9.NewValue other_inst_plans, a10.NewValue housing, a11.NewValue job_type, a12.NewValue telephone, a13.NewValue foreign_worker, a1.duration_months,  a1.credit_amount, a1.installment_rate,  a1.present_res_since, a1.age, a1.num_existing_credits,  a1.num_dependents, (a1.credit_rating - 1) credit_rating
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

// MAGIC %md #### Split prepared data into two buckets: **good credit** and **bad credit**

// COMMAND ----------

prepared_data.cache()
/*prepared_data.registerTempTable("prepared_data")
val good_credit = prepared_data.filter(r => r(20).asInstanceOf[Int] == 1).map(s => s.slice(0, 19))
val bad_credit = prepared_data.filter(r => r(20).asInstanceOf[Int] == 0).map(s => s.slice(0, 19))
good_credit.cache()
bad_credit.cache()
good_credit.take(5).foreach(println)
bad_credit.take(5).foreach(println)*/

// COMMAND ----------

//%sql desc prepared_data