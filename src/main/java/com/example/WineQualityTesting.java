package com.example;

import org.apache.spark.SparkConf; 
import org.apache.spark.api.java.JavaRDD; 
import org.apache.spark.api.java.JavaSparkContext; 
import org.apache.spark.ml.feature.VectorAssembler; 
import org.apache.spark.ml.linalg.Vector; 
import org.apache.spark.mllib.evaluation.MulticlassMetrics; 
import org.apache.spark.mllib.linalg.Vectors; 
import org.apache.spark.mllib.regression.LabeledPoint; // Represents labeled data points for ML models
import org.apache.spark.mllib.tree.model.RandomForestModel; // Spark MLlib Random Forest model
import org.apache.spark.sql.Dataset; // Represents structured data in Spark
import org.apache.spark.sql.Row; // Represents a single row of data
import org.apache.spark.sql.SparkSession; // Entry point for Spark SQL
import scala.Tuple2; // Represents a tuple with two elements

public class WineQualityTesting {

    public static void main(String[] args) {
        // Step 1: Initialize Spark configuration and context
        SparkConf conf = new SparkConf()
                .setAppName("WineQualityTesting") // Application name
                .setMaster("local"); // Local execution for testing
        JavaSparkContext sc = new JavaSparkContext(conf); // Initialize JavaSparkContext
        SparkSession spark = SparkSession.builder() // Initialize SparkSession
                .config(conf)
                .getOrCreate();

        // Step 2: Verify command-line arguments
        if (args.length < 1) { // Ensure the test dataset path is provided as an argument
            System.err.println("Usage: Testing <test_dataset_path>");
            System.exit(1); // Exit the program if no argument is provided
        }
        String testDatasetPath = args[0]; // Store the test dataset path

        // Step 3: Load the test dataset
        Dataset<Row> testDataset = spark.read()
                .format("csv") // Input format: CSV
                .option("header", "true") // CSV has a header row
                .option("sep", ";") // Use semicolon as the delimiter
                .load(testDatasetPath); // Load the dataset from the provided path

        System.out.println("Schema of the test dataset:");
        testDataset.printSchema(); // Display the schema of the test dataset
        testDataset.show(); // Show a sample of the test dataset

        // Step 4: Preprocess the dataset
        // Convert non-'quality' columns to float and rename 'quality' to 'label' for ML compatibility
        for (String colName : testDataset.columns()) {
            if (!colName.equals("quality")) { // Skip the 'quality' column
                testDataset = testDataset.withColumn(
                        colName,
                        testDataset.col(colName).cast("float") // Cast to float for numeric processing
                );
            }
        }
        testDataset = testDataset.withColumnRenamed("quality", "label"); // Rename 'quality' to 'label'

        // Step 5: Assemble features into a single vector column
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(testDataset.columns()) // Use all columns as input features
                .setOutputCol("features"); // Output column name: 'features'

        Dataset<Row> transformedTestData = assembler.transform(testDataset) // Apply feature assembly
                .select("features", "label"); // Select only 'features' and 'label' columns

        System.out.println("Transformed test dataset:");
        transformedTestData.show(); // Display the transformed dataset

        // Step 6: Convert DataFrame to RDD of LabeledPoint
        // Each LabeledPoint contains a label and its corresponding feature vector
        JavaRDD<LabeledPoint> testRDD = convertToLabeledPointRDD(sc, transformedTestData);

        // Step 7: Load the pre-trained Random Forest model
        String modelPath = "s3://winesparkbucket/TrainingDataset.model/"; // S3 path to the saved model
        RandomForestModel randomForestModel = RandomForestModel.load(sc.sc(), modelPath); // Load the model
        System.out.println("Model successfully loaded from: " + modelPath);

        // Step 8: Perform predictions using the model
        JavaRDD<Double> predictions = randomForestModel.predict(testRDD.map(LabeledPoint::features)); // Predict labels

        // Step 9: Map predictions with actual labels
        // Create an RDD of tuples containing (actual label, predicted label)
        JavaRDD<Tuple2<Double, Double>> labelsAndPredictions = testRDD.map(lp ->
                new Tuple2<>(lp.label(), randomForestModel.predict(lp.features()))
        );

        // Step 10: Convert labels and predictions to a DataFrame
        Dataset<Row> predictionsDF = spark.createDataFrame(labelsAndPredictions, Tuple2.class) // Convert RDD to DataFrame
                .toDF("label", "prediction"); // Rename columns for clarity

        System.out.println("Predictions DataFrame:");
        predictionsDF.show(); // Display the DataFrame of predictions

        // Step 11: Evaluate model performance
        MulticlassMetrics metrics = new MulticlassMetrics(labelsAndPredictions.rdd()); // Initialize evaluation metrics
        System.out.println("Evaluation Metrics:");
        // For weighted F1-score
        System.out.println("Weighted F1-score: " + metrics.weightedFMeasure());

        System.out.println("Confusion Matrix:\n" + metrics.confusionMatrix()); // Print confusion matrix
        System.out.println("Weighted Precision: " + metrics.weightedPrecision()); // Print weighted precision
        System.out.println("Weighted Recall: " + metrics.weightedRecall()); // Print weighted recall
        System.out.println("Accuracy: " + metrics.accuracy()); // Print accuracy

        // Step 12: Calculate test error
        long incorrectPredictions = labelsAndPredictions.filter(lp -> !lp._1().equals(lp._2())).count(); // Count mismatches
        double testError = (double) incorrectPredictions / testRDD.count(); // Calculate error rate
        System.out.println("Test Error: " + testError);

        // Step 13: Close Spark context
        sc.close(); // Release Spark resources
    }

    // Helper function to convert a DataFrame to RDD<LabeledPoint>
    private static JavaRDD<LabeledPoint> convertToLabeledPointRDD(JavaSparkContext sc, Dataset<Row> df) {
        return df.toJavaRDD().map(row -> {
            double label = row.getDouble(row.fieldIndex("label")); // Extract label
            org.apache.spark.ml.linalg.Vector mlVector = row.getAs("features"); // Extract features (ML Vector)
            org.apache.spark.mllib.linalg.Vector mllibVector = org.apache.spark.mllib.linalg.Vectors.fromML(mlVector); // Convert to MLlib vector
            return new LabeledPoint(label, mllibVector); // Return LabeledPoint
        });
    }
    
    
}
