from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark session
spark = SparkSession.builder \
    .appName('PredictWineQuality') \
    .master('local[*]') \
    .getOrCreate()

# S3 paths
model_path = "s3a://pa2winequalitybucket/optimized_wine_model_rf.model"
validation_dataset_path = "s3a://pa2winequalitybucket/ValidationDataset.csv"

# Load the saved pipeline model (contains all preprocessing steps)
model_pipeline = PipelineModel.load(model_path)

# Load and clean validation data
raw_val_data = spark.read.csv(validation_dataset_path, header=True, inferSchema=True, sep=';')

def clean_column_names(df):
    cleaned_columns = [col_name.strip('"').strip() for col_name in df.columns]
    return df.toDF(*cleaned_columns)

raw_val_data = clean_column_names(raw_val_data)

# Add the `alcohol_density_ratio` feature
raw_val_data = raw_val_data.withColumn("alcohol_density_ratio", col("alcohol") / col("density"))

# Drop unnecessary columns and convert `quality` to binary labels
columns_to_drop = ['residual sugar', 'free sulfur dioxide']
raw_val_data = raw_val_data.drop(*columns_to_drop)
raw_val_data = raw_val_data.withColumn("label", when(col("quality") >= 7, 1).otherwise(0))

# Use the trained pipeline for preprocessing and prediction
predictions = model_pipeline.transform(raw_val_data)

# Evaluate predictions
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1_score = evaluator.evaluate(predictions)

# Evaluate accuracy
accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = accuracy_evaluator.evaluate(predictions)

# Evaluate Weighted Recall
evaluator_weighted_recall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
weighted_recall = evaluator_weighted_recall.evaluate(predictions)

# Display evaluation metrics
print("==========================================")
print(f"Validation Accuracy: {accuracy}")
print(f"Validation F1 Score: {f1_score}")
print(f"Validation Weighted Recall: {weighted_recall}")
print("==========================================")

# Stop Spark session
spark.stop()
