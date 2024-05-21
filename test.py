import pyspark
from pyspark import SQLContext
from pyspark.sql import SparkSession
spark = SparkSession.builder.config("spark.driver.host","localhost").appName("testing").getOrCreate()
conf = pyspark.SparkConf()
spark_context = SparkSession.builder.config(conf=conf).getOrCreate()
from pyspark.sql import functions as F
from pyspark.sql.window import Window

claims = spark.read.options(header=True).csv(r"C:\Users\kring\test\DE1_0_2008_to_2010_Outpatient_Claims_Sample_20.csv")
summary = spark.read.options(header=True).csv(r"C:\Users\kring\test\DE1_0_2009_Beneficiary_Summary_File_Sample_20.csv")

# Convert chronic illness cols to single, concatenating multiple diagnoses

diagnoses_cols = ["SP_ALZHDMTA", "SP_CHF", "SP_CHRNKIDN", "SP_CNCR", "SP_COPD", "SP_DEPRESSN", "SP_DIABETES", "SP_ISCHMCHT", "SP_OSTEOPRS", "SP_RA_OA", "SP_STRKETIA"]
new_summary = summary

for column in new_summary.columns:
    if column in diagnoses_cols:
        new_summary = new_summary.withColumn(column, F.when(new_summary[column] == "2", "0").otherwise(new_summary[column]))

new_summary = new_summary.withColumn("Conditions_Total", F.expr("+".join(diagnoses_cols)))
new_summary.show(5, truncate=False, vertical=True)

# If a member has 3 or more diagnoses, categorize as multiple

new_summary = new_summary.withColumn("Conditions_Total", F.when(F.col("Conditions_Total") >= 3, "Multiple").otherwise(F.col("Conditions_Total")))
new_summary.show(5, truncate=False, vertical=True)

# Join claims and benefit data

joined = new_summary.join(claims, "DESYNPUF_ID", "full")
joined.show(5, truncate=False, vertical=True)

# What is the distribution of races

df = joined.withColumn("BENE_RACE_CD", F.when(F.col("BENE_RACE_CD") == "1", "White").otherwise(F.col("BENE_RACE_CD")))
df = df.withColumn("BENE_RACE_CD", F.when(F.col("BENE_RACE_CD") == "2", "Black").otherwise(F.col("BENE_RACE_CD")))
df = df.withColumn("BENE_RACE_CD", F.when(F.col("BENE_RACE_CD") == "3", "Others").otherwise(F.col("BENE_RACE_CD")))
df = df.withColumn("BENE_RACE_CD", F.when(F.col("BENE_RACE_CD") == "5", "Hispanic").otherwise(F.col("BENE_RACE_CD")))
df.select("BENE_RACE_CD").groupBy("BENE_RACE_CD").count().orderBy("count", ascending=False).show()

# What is the most common chronic illness combination

df3b = df.select("DESYNPUF_ID", *diagnoses_cols)
df3b = df3b.withColumn("SP_ALZHDMTA", F.when(F.col("SP_ALZHDMTA") == "1", "Alzheimer").otherwise(None))
df3b = df3b.withColumn("SP_CHF", F.when(F.col("SP_CHF") == "1", "Heart Failure").otherwise(None))
df3b = df3b.withColumn("SP_CHRNKIDN", F.when(F.col("SP_CHRNKIDN") == "1", "Chronic Kidney Disease").otherwise(None))
df3b = df3b.withColumn("SP_CNCR", F.when(F.col("SP_CNCR") == "1", "Cancer").otherwise(None))
df3b = df3b.withColumn("SP_COPD", F.when(F.col("SP_COPD") == "1", "COPD").otherwise(None))
df3b = df3b.withColumn("SP_DEPRESSN", F.when(F.col("SP_DEPRESSN") == "1", "Depression").otherwise(None))
df3b = df3b.withColumn("SP_DIABETES", F.when(F.col("SP_DIABETES") == "1", "Diabetes").otherwise(None))
df3b = df3b.withColumn("SP_ISCHMCHT", F.when(F.col("SP_ISCHMCHT") == "1", "Ischemic Heart Disease").otherwise(None))
df3b = df3b.withColumn("SP_OSTEOPRS", F.when(F.col("SP_OSTEOPRS") == "1", "Osteoporosis").otherwise(None))
df3b = df3b.withColumn("SP_RA_OA", F.when(F.col("SP_RA_OA") == "1", "RA/OA").otherwise(None))
df3b = df3b.withColumn("SP_STRKETIA", F.when(F.col("SP_STRKETIA") == "1", "Stroke").otherwise(None))
df3b = df3b.distinct()

result = df3b \
          .withColumn("temp", F.array(*diagnoses_cols)) \
            .withColumn("Illness_List", F.expr("FILTER(temp, x -> x is not null)")) \
                .drop("temp")

df3b = result.select("DESYNPUF_ID", "Illness_List").distinct()
df3b.select("Illness_List").groupBy("Illness_List").count().orderBy("count", ascending=False).show(10, truncate=False)
rejoined = df.join(df3b, "DESYNPUF_ID", "left")
rejoined = rejoined.distinct()

# Which chronic illness combo has highest total cost

cost_cols = ["MEDREIMB_IP", "BENRES_IP", "PPPYMT_IP", "MEDREIMB_OP", "BENRES_OP", "PPPYMT_OP", "MEDREIMB_CAR", "BENRES_CAR", "PPPYMT_CAR"]
df3c = rejoined.select("DESYNPUF_ID", *cost_cols, "Illness_List").distinct()
df3c = df3c.withColumn('Total_Cost', F.expr('+'.join(cost_cols)))
df3c.groupBy("Illness_List").agg(F.sum("Total_Cost")).orderBy("sum(Total_Cost)", ascending=False).show(10, truncate=False)

# Which chronic illness combo has highest cost per member

df3d = df3c.select("DESYNPUF_ID", "Illness_List", "Total_Cost")
w = Window.partitionBy("Illness_List")
df3d = df3d.withColumn("Total_Cost_Sum", F.sum("Total_Cost").over(w))
df3d = df3d.withColumn("Illness_Count", F.count("Illness_List").over(w))
df3d.groupby("Illness_List", "Total_Cost_Sum", "Illness_Count") \
    .agg((F.col("Total_Cost_Sum") / F.col("Illness_Count")).alias("Cost_Per_Mem")) \
        .orderBy("Cost_Per_Mem", ascending=False).show(20, truncate=False, vertical=True)

spark.stop()