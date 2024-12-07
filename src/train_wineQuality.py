from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# Optimized Spark session
spark = SparkSession.builder \
    .appName('TrainWineQuality') \
    .config("spark.dynamicAllocation.enabled", "true") \
    .getOrCreate()

# S3 paths
train_dataset_path = "s3a://pa2winequalitybucket/TrainingDataset.csv"
output_model_path = "s3a://pa2winequalitybucket/optimized_wine_model_rf.model"

# Load dataset
raw_train_data = spark.read.csv(train_dataset_path, header=True, inferSchema=True, sep=';')

# Clean column names
def clean_and_preprocess(df):
    cleaned_columns = [col_name.strip('"').strip() for col_name in df.columns]
    return df.toDF(*cleaned_columns)

raw_train_data = clean_and_preprocess(raw_train_data)

# Feature Engineering: Add interaction term
raw_train_data = raw_train_data.withColumn("alcohol_density_ratio", col("alcohol") / col("density"))

# Drop less important features (initial step)
columns_to_drop = ['residual sugar', 'free sulfur dioxide']
raw_train_data = raw_train_data.drop(*columns_to_drop)

# Convert `quality` to binary classification: good wine (1), bad wine (0)
raw_train_data = raw_train_data.withColumn("label", when(col("quality") >= 7, 1).otherwise(0))

# Oversampling for Class Balancing
def balance_classes(df):
    major_df = df.filter(df["label"] == 1)
    minor_df = df.filter(df["label"] == 0)
    oversampled_minor_df = minor_df.sample(withReplacement=True, fraction=major_df.count() / minor_df.count())
    return major_df.union(oversampled_minor_df)

balanced_train_data = balance_classes(raw_train_data)

# Feature Selection
feature_columns = [
    "fixed acidity", "volatile acidity", "citric acid", 
    "chlorides", "total sulfur dioxide", "density", 
    "pH", "sulphates", "alcohol", "alcohol_density_ratio"
]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)

# Random Forest Classifier
rf = RandomForestClassifier(featuresCol="scaledFeatures", labelCol="label", seed=42)

# Hyperparameter Grid
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [50, 100, 200]) \
    .addGrid(rf.maxDepth, [10, 15]) \
    .addGrid(rf.maxBins, [32]) \
    .addGrid(rf.minInstancesPerNode, [1, 2]) \
    .build()

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

cross_validator = CrossValidator(estimator=Pipeline(stages=[assembler, scaler, rf]),
                                  estimatorParamMaps=paramGrid,
                                  evaluator=evaluator,
                                  numFolds=5)

# Train the model
print("Training the model with specified paramGrid...")
cv_model = cross_validator.fit(balanced_train_data)

# Evaluate additional metrics on training data
predictions = cv_model.transform(balanced_train_data)

# Evaluate Weighted Recall
evaluator_weighted_recall = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="weightedRecall"
)
weighted_recall = evaluator_weighted_recall.evaluate(predictions)

# Display Confusion Matrix
print("Confusion Matrix:")
predictions.groupBy("label", "prediction").count().show()

# Save the best model
cv_model.bestModel.write().overwrite().save(output_model_path)
print("=================================================================================")
print(f"Model successfully saved to {output_model_path}")
print(f"Training Weighted Recall: {weighted_recall}")
print("=================================================================================")

# Stop Spark session
spark.stop()
