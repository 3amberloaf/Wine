package com.example;

import java.util.HashMap;
import java.util.Map;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RandomForestTrainer {
    private static final Logger logger = LoggerFactory.getLogger(RandomForestTrainer.class);

    public static void main(String[] args) {
        logger.info("Starting RandomForestTrainer...");

        // Configure Spark
        SparkConf conf = new SparkConf().setAppName("RandomForestTrainer").setMaster("spark://172.31.31.175:7077");
        JavaSparkContext sc = new JavaSparkContext(conf);
        SparkSession spark = SparkSession.builder().config(conf).getOrCreate();

        logger.info("Spark context initialized.");

        // Path to the training dataset
        String trainingPath = args[0];
        String modelPath = "/home/ubuntu/models/random-forest-model";

        // Load the training dataset
        Dataset<Row> trainingData = spark.read()
                .format("csv")
                .option("header", "true")
                .option("sep", ";")
                .load(trainingPath);

        logger.info("Training data loaded from path: {}", trainingPath);
        trainingData.printSchema();
        trainingData.show(5);

        // Clean up column names to remove unexpected characters and normalize
        String[] originalColumns = trainingData.columns();
        String[] cleanedColumns = new String[originalColumns.length];

        // Generate cleaned column names by removing extra quotes
        for (int i = 0; i < originalColumns.length; i++) {
            cleanedColumns[i] = originalColumns[i].replaceAll("\"", "").trim(); // Clean the column names
        }

        // Rename columns in the DataFrame to use the cleaned column names
        for (int i = 0; i < originalColumns.length; i++) {
            trainingData = trainingData.withColumnRenamed(originalColumns[i], cleanedColumns[i]);
        }

        logger.info("Column names cleaned and normalized.");

        // Ensure all feature columns are cast to float and rename 'quality' to 'label'
        for (String colName : trainingData.columns()) {
            if (!colName.equals("quality")) {
                trainingData = trainingData.withColumn(colName, trainingData.col(colName).cast("float"));
            }
        }

        // Check if the 'quality' column exists after cleaning column names
        if (!java.util.Arrays.asList(trainingData.columns()).contains("quality")) {
            logger.error("Column 'quality' not found in dataset after normalization. Exiting.");
            sc.close();
            return;
        }

        // Rename 'quality' to 'label' and drop it from the DataFrame
        trainingData = trainingData.withColumn("label", trainingData.col("quality").cast("double"))
                                   .drop("quality");

        logger.info("Columns cast to appropriate data types and 'quality' renamed to 'label'.");
        trainingData.printSchema();

        // Assemble features into a single vector column
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{
                        "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
                        "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide",
                        "density", "pH", "sulphates", "alcohol"
                })
                .setOutputCol("features");

        Dataset<Row> assembledData = assembler.transform(trainingData).select("features", "label");

        logger.info("Features assembled into a single vector column.");
        assembledData.printSchema();
        assembledData.show(5);

        // Convert DataFrame to JavaRDD<LabeledPoint>
        JavaRDD<LabeledPoint> labeledPoints = toLabeledPoint(sc, assembledData);

        logger.info("Data converted to JavaRDD<LabeledPoint>.");

        // Define the categorical features info as a Java Map (empty map in this case)
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();

        // Train the Random Forest model
        RandomForestModel model = RandomForest.trainClassifier(
                labeledPoints,            // JavaRDD
                10,                       // Number of classes
                categoricalFeaturesInfo,  // Categorical features map
                100,                      // Number of trees
                "auto",                   // Feature subset strategy
                "gini",                   // Impurity
                10,                       // Max depth
                32,                       // Max bins
                12345                     // Random seed
        );

        logger.info("Random Forest model trained successfully.");

        // Save the trained model
        try {
            model.save(sc.sc(), modelPath);
            logger.info("Model saved successfully to path: {}", modelPath);
        } catch (Exception e) {
            logger.error("Error saving model to path: {}", modelPath, e);
        }

        // Close Spark context
        sc.close();
        logger.info("Spark context closed. Exiting RandomForestTrainer.");
    }

    private static JavaRDD<LabeledPoint> toLabeledPoint(JavaSparkContext sc, Dataset<Row> df) {
        return df.toJavaRDD().map(row -> {
            double label = row.getDouble(row.fieldIndex("label"));
            org.apache.spark.ml.linalg.Vector mlVector = row.getAs("features");
            org.apache.spark.mllib.linalg.Vector mllibVector = org.apache.spark.mllib.linalg.Vectors.fromML(mlVector);
            return new LabeledPoint(label, mllibVector);
        });
    }
}
