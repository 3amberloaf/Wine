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

public class RandomForestTrainer {
    public static void main(String[] args) {
        // Configure Spark
        SparkConf conf = new SparkConf().setAppName("RandomForestTrainer").setMaster("spark://172.31.31.175:7077");
        JavaSparkContext sc = new JavaSparkContext(conf);
        SparkSession spark = SparkSession.builder().config(conf).getOrCreate();

        // Path to the training dataset
        String trainingPath = args[0];
        String modelPath = "/home/ubuntu/models/random-forest-model";



        // Load the training dataset
        Dataset<Row> trainingData = spark.read()
                .format("csv")
                .option("header", "true")
                .option("sep", ";")
                .load(trainingPath);

                trainingData.printSchema();
                trainingData.show(5); 
                

        // Convert all feature columns except 'quality' to float and rename 'quality' to 'label'
        for (String colName : trainingData.columns()) {
            if (!colName.equals("quality")) {
                trainingData = trainingData.withColumn(colName, trainingData.col(colName).cast("float"));
            }
        }
        trainingData = trainingData.withColumn("label", trainingData.col("quality").cast("double"))
                                .drop("quality");


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

        assembledData.printSchema();
        assembledData.show(5); // Inspect first 5 rows

        // Convert DataFrame to JavaRDD<LabeledPoint>
        JavaRDD<LabeledPoint> labeledPoints = toLabeledPoint(sc, assembledData);

        // Define the categorical features info as a Java Map (empty map in this case)
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>(); // Example: Specify categorical features if any

        // Train the Random Forest model
        RandomForestTrainer model = RandomForestTrainer.trainClassifier(
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

        // Save the trained model
        model.save(sc.sc(), modelPath);
        System.out.println("Model trained and saved successfully to: " + modelPath);

        // Close Spark context
        sc.close();
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
