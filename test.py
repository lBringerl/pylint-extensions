from pyspark import SparkContext
import pyspark


def function():
    with SparkContext.getOrCreate() as spark:
        spark.table().some_attribute.some_attr_call().toPandas()


with pyspark.SparkContext.getOrCreate() as spark:
    spark.DataFrame().repartition()
