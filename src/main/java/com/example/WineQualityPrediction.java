package com.example;

import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;


public class WineQualityPrediction {
    public static void main(String[] args) {
        // 1. Set up SparkSession
        SparkSession spark = SparkSession.builder()
                .appName("Wine Quality Prediction")
                .master("local[*]") // Change to cluster mode for EC2
                .getOrCreate();

        // 2. Load the Trained Model
        String modelPath = "s3://your-model-path/spark-model";
        PipelineModel model = PipelineModel.load(modelPath);

        // 3. Load the Input Dataset
        String testFilePath = "s3://your-dataset-path/ValidationDataset.csv";
        Dataset<Row> testData = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(testFilePath);

        // 4. Make Predictions
        Dataset<Row> predictions = model.transform(testData);

        // 5. Evaluate the Model Performance
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label") // Adjust based on your dataset's label column
                .setPredictionCol("prediction")
                .setMetricName("f1");

        double f1Score = evaluator.evaluate(predictions);
        System.out.println("F1 Score = " + f1Score);

        // Stop Spark Session
        spark.stop();
    }
}
