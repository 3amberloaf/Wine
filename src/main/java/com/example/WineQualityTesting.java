package com.example;

import org.apache.spark.SparkConf; 
import org.apache.spark.api.java.JavaRDD; 
import org.apache.spark.api.java.JavaSparkContext; 
import org.apache.spark.ml.feature.VectorAssembler; 
import org.apache.spark.mllib.evaluation.MulticlassMetrics; 
import org.apache.spark.mllib.regression.LabeledPoint; 
import org.apache.spark.mllib.tree.model.RandomForestModel; 
import org.apache.spark.sql.Dataset; 
import org.apache.spark.sql.Row; 
import org.apache.spark.sql.SparkSession; 
import scala.Tuple2; 

public class WineQualityTesting {

    public static void main(String[] args) {
        // Initialize Spark configuration 
        SparkConf conf = new SparkConf()
                .setAppName("WineQualityTesting") 
                .setMaster("local"); // Local execution for testing
        JavaSparkContext sc = new JavaSparkContext(conf); 
        SparkSession spark = SparkSession.builder() 
                .config(conf)
                .getOrCreate();

        if (args.length < 1) { // Ensure the test dataset path is provided as an argument
            System.err.println("Usage: Testing <test_dataset_path>");
            System.exit(1); // Exit the program if no argument is provided
        }
        String testDatasetPath = args[0]; // Store the test dataset path

        // Load the test dataset
        Dataset<Row> testDataset = spark.read()
                .format("csv") 
                .option("header", "true") 
                .option("sep", ";") 
                .load(testDatasetPath); 

        System.out.println("Schema of the test dataset:");
        testDataset.printSchema(); 
        testDataset.show(); 

        // Preprocess the dataset
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

        // Assemble features into a single vector column
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(testDataset.columns()) // Use all columns as input features
                .setOutputCol("features"); // Output column name: 'features'

        Dataset<Row> transformedTestData = assembler.transform(testDataset) // Apply feature assembly
                .select("features", "label"); // Select only 'features' and 'label' columns

        System.out.println("Transformed test dataset:");
        transformedTestData.show(); 

        // Convert DataFrame to RDD of LabeledPoint
        // Each LabeledPoint contains a label and its corresponding feature vector
        JavaRDD<LabeledPoint> testRDD = convertToLabeledPointRDD(sc, transformedTestData);

        // Load the pre-trained Random Forest model
        String modelPath = "s3://winesparkbucket/training.model/"; // S3 path to the saved model
        RandomForestModel randomForestModel = RandomForestModel.load(sc.sc(), modelPath); // Load the model
        System.out.println("Model successfully loaded from: " + modelPath);

        // Step 8: Perform predictions using the model
        JavaRDD<Double> predictions = randomForestModel.predict(testRDD.map(LabeledPoint::features)); // Predict labels

        // Map predictions with actual labels
        // Create an RDD of tuples containing (actual label, predicted label)
        JavaRDD<Tuple2<Double, Double>> labelsAndPredictions = testRDD.map(lp ->
                new Tuple2<>(lp.label(), randomForestModel.predict(lp.features()))
        );

        // Convert labels and predictions to a DataFrame
        Dataset<Row> predictionsDF = spark.createDataFrame(labelsAndPredictions, Tuple2.class) // Convert RDD to DataFrame
                .toDF("label", "prediction"); 

        System.out.println("Predictions DataFrame:");
        predictionsDF.show(); 

        // Evaluate model performance
        MulticlassMetrics metrics = new MulticlassMetrics(labelsAndPredictions.rdd()); 
        System.out.println("Evaluation Metrics:");
        // For weighted F1-score
        System.out.println("Weighted F1-score: " + metrics.weightedFMeasure());

        System.out.println("Confusion Matrix:\n" + metrics.confusionMatrix()); 
        System.out.println("Weighted Precision: " + metrics.weightedPrecision()); 
        System.out.println("Weighted Recall: " + metrics.weightedRecall()); 
        System.out.println("Accuracy: " + metrics.accuracy()); 


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
