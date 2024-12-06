package com.example;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Arrays;
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

public class TrainModel {
    private static final Logger logger = LoggerFactory.getLogger(TrainModel.class);

    public static void main(String[] args) {
        logger.info("Starting TrainModel...");


        // Configure Spark app
        SparkConf conf = new SparkConf()
            .setAppName("TrainModel")
            .setMaster("local[*]")
            .set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem");

            // Initialize Spark context and session
            JavaSparkContext sc = new JavaSparkContext(conf);
            SparkSession spark = SparkSession.builder().config(conf).getOrCreate();
    
            logger.info("Spark context initialized.");
    
            String trainingPath = "s3a://winesparkbucket/TrainingDataset.csv";
            String modelPath = "s3a://winesparkbucket/random-forest-model";
    
            // Load the training dataset from s3
            Dataset<Row> trainingData = spark.read()
                    .format("csv")
                    .option("header", "true")
                    .option("sep", ";")
                    .option("inferSchema", "true")
                    .load(trainingPath);
    
            logger.info("Training data loaded from path: {}", trainingPath);
            trainingData.printSchema();
            trainingData.show(5);
    
            // Clean the column names to remove extra quotes
            String[] originalColumns = trainingData.columns();
            String[] newColumns = Arrays.stream(originalColumns)
                                        .map(name -> name.replaceAll("\"", "").trim())
                                        .toArray(String[]::new);

            // Rename columns in the dataset to use the cleaned column names
            for (int i = 0; i < originalColumns.length; i++) {
                trainingData = trainingData.withColumnRenamed(originalColumns[i], newColumns[i]);
            }
    
            // Identify label and feature columns dynamically
            String labelColumn = "quality"; 
            List<String> featureColumns = new ArrayList<>();
            for (String column : newColumns) {
                if (!column.equals(labelColumn)) {
                    featureColumns.add(column);
                }
            }
    
            // Make sure label column is there
            if (!Arrays.asList(newColumns).contains(labelColumn)) {
                logger.error("Label column '{}' not found. Exiting.", labelColumn);
                sc.close();
                return;
            }
    
            for (String colName : featureColumns) {
                trainingData = trainingData.withColumn(colName, trainingData.col(colName).cast("float"));
            }
            trainingData = trainingData.withColumn("label", trainingData.col(labelColumn).cast("double"))
                                       .drop(labelColumn);
    
            logger.info("Columns cast to appropriate types and 'quality' renamed to 'label'.");
            trainingData.printSchema();
    
            // Assemble features into a single vector column
            VectorAssembler assembler = new VectorAssembler()
                    .setInputCols(featureColumns.toArray(new String[0]))
                    .setOutputCol("features");
    
            Dataset<Row> assembledData = assembler.transform(trainingData).select("features", "label");
    
            logger.info("Features assembled.");
            assembledData.printSchema();
            assembledData.show(5);
    
            // Convert DataFrame to JavaRDD<LabeledPoint> for training
            JavaRDD<LabeledPoint> labeledPoints = toLabeledPoint(sc, assembledData);
    
            logger.info("Data converted to JavaRDD<LabeledPoint>.");
    
            // Define the categorical features info
            Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
    
            // Train the Random Forest model
            RandomForestModel model = RandomForest.trainClassifier(
                    labeledPoints,            // JavaRDD
                    10,                       
                    categoricalFeaturesInfo,  
                    100,                     
                    "auto",                 
                    "gini",                   
                    10,                      
                    32,                     
                    12345                   
            );
    
            logger.info("Random Forest model trained successfully.");
    
            // Save the trained model to s3
            try {
                model.save(sc.sc(), modelPath);
                logger.info("Model saved successfully to path: {}", modelPath);
            } catch (Exception e) {
                logger.error("Error saving model to path: {}", modelPath, e);
            }
    
            // Close Spark context
            sc.close();
            logger.info("Spark context closed. Exiting TrainModel.");
        }
        
        // Converts a Dataset<Row> into an RDD of LabeledPoint for MLlib compatibility
        private static JavaRDD<LabeledPoint> toLabeledPoint(JavaSparkContext sc, Dataset<Row> df) {
            return df.toJavaRDD().map(row -> {
                double label = row.getDouble(row.fieldIndex("label"));
                org.apache.spark.ml.linalg.Vector mlVector = row.getAs("features");
                org.apache.spark.mllib.linalg.Vector mllibVector = org.apache.spark.mllib.linalg.Vectors.fromML(mlVector);
                return new LabeledPoint(label, mllibVector);
            });
        }
    }