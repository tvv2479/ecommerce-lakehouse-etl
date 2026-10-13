"""
E-Commerce Big Data Lakehouse Pipeline (PySpark + Delta Lake)
Описание: Демонстрация полного ETL-цикла по архитектуре Medallion 
(Bronze -> Silver -> Gold) с использованием локального хранилища (MinIO / S3).
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, StringType, 
    LongType, DoubleType, TimestampType, DateType
)
from delta.tables import DeltaTable
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EcommerceETL")
logger.info(">>> Запуск этапа очистки данных...")

def create_spark_session():
    """
    Создание SparkSession с поддержкой Delta Lake и S3 (MinIO)
    """
    return SparkSession.builder \
        .appName("EcommerceLakehouseETL") \
        .master("local[*]") \
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.1.0,org.apache.hadoop:hadoop-aws:3.3.4") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.hadoop.fs.s3a.endpoint", "http://localhost:9000") \
        .config("spark.hadoop.fs.s3a.access.key", "minioadmin") \
        .config("spark.hadoop.fs.s3a.secret.key", "minioadmin") \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()


def generate_mock_data(spark):
    """Генерация тестовых данных для демонстрации"""
    print(">>> Генерация фейковых данных для демонстрации...")
    
    # 1. Сырые клики пользователей
    clicks_data = [
        ("c1", 101, "click", "2024-10-01 10:00:00"),
        ("c2", 102, "click", "2024-10-01 10:05:00"),
        ("c1", 101, "click", "2024-10-01 10:00:00"), # дубликат
        ("c3", 103, "view",  "2024-10-01 10:10:00"),
        ("c4", 104, "buy",   "2024-10-01 10:15:00"),
    ]
    clicks_df = spark.createDataFrame(clicks_data, ["event_id", "user_id", "action", "event_time"])
    clicks_df.write.mode("overwrite").json("s3a://ecommerce/raw/clicks/")

    # 2. Справочник категорий товаров
    categories_data = [
        (1, "Electronics"),
        (2, "Clothing"),
        (3, "Home & Kitchen")
    ]
    cat_df = spark.createDataFrame(categories_data, ["category_id", "category_name"])
    cat_df.write.mode("overwrite").parquet("s3a://ecommerce/raw/categories/")


def run_etl():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    # Генерирация тестовых данных (в реальном проекте они поступают извне)
    generate_mock_data(spark)

    logger.info(" ЭТАП 1: BRONZE LAYER (Сырые данные -> Delta)")
    
    # Читение сырых JSON-логов
    raw_clicks_df = spark.read.json("s3a://ecommerce/raw/clicks/")
    
    # Записываем в Bronze слой без изменений (только добавляем служебные поля)
    raw_clicks_df \
        .withColumn("ingested_at", F.current_timestamp()) \
        .write \
        .format("delta") \
        .mode("overwrite") \
        .save("s3a://ecommerce/bronze/clicks")
    
    logger.info("Данные успешно записаны в Bronze слой.")
    logger.info(" ЭТАП 2: SILVER LAYER (Очистка и дедупликация)")
    
    # Читаем из Bronze
    bronze_df = spark.read.format("delta").load("s3a://ecommerce/bronze/clicks")
    
    # Трансформации: приведение типов, фильтрация, дедупликация по event_id
    window_spec = Window.partitionBy("event_id").orderBy(F.col("event_time").desc())
    
    silver_df = bronze_df \
        .withColumn("event_time", F.to_timestamp("event_time")) \
        .withColumn("event_date", F.to_date("event_time")) \
        .withColumn("row_num", F.row_number().over(window_spec)) \
        .filter(F.col("row_num") == 1) \
        .drop("row_num") \
        .filter(F.col("user_id").isNotNull())

    # Записываем в Silver слой с партиционированием по дате
    silver_df \
        .write \
        .format("delta") \
        .mode("overwrite") \
        .partitionBy("event_date") \
        .save("s3a://ecommerce/silver/clicks")
        
    logger.info("Данные очищены, дедуплицированы и записаны в Silver слой.")
    logger.info(" ЭТАП 3: GOLD LAYER (Бизнес-витрины и Broadcast Join)")
    
    # Читаем очищенные клики и справочник категорий
    silver_clicks = spark.read.format("delta").load("s3a://ecommerce/silver/clicks")
    raw_categories = spark.read.parquet("s3a://ecommerce/raw/categories/")

    # Имитируем добавление category_id к событиям для примера соединения
    clicks_with_cat = silver_clicks.withColumn("category_id", (F.col("user_id") % 3) + 1)

    # Демонстрация Broadcast Hash Join (соединение большой таблицы с маленьким справочником)
    gold_agg_df = clicks_with_cat.join(
        F.broadcast(raw_categories),
        on="category_id",
        how="inner"
    ).groupBy("event_date", "category_name", "action").agg(
        F.count("user_id").alias("total_actions"),
        F.countDistinct("user_id").alias("unique_users")
    )

    # Записываем витрину в Gold слой
    gold_agg_df \
        .write \
        .format("delta") \
        .mode("overwrite") \
        .save("s3a://ecommerce/gold/daily_activity_mart")

    logger.info("Бизнес-витрина успешно рассчитана и записана в Gold слой.")
    logger.info("РЕЗУЛЬТАТ (Чтение готовой витрины из Gold):")
    
    result_df = spark.read.format("delta").load("s3a://ecommerce/gold/daily_activity_mart")
    result_df.show(truncate=False)

    spark.stop()


if __name__ == "__main__":
    run_etl()