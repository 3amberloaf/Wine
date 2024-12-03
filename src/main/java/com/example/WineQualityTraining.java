package com.example;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Arrays;
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

public class WineQualityTraining {
    public static void main(String[] args) {
        // Configure Spark
        SparkConf conf = new SparkConf().setAppName("WineQualityTrainer").setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);
        SparkSession spark = SparkSession.builder().config(conf).getOrCreate();

        try {
            // Paths
            String inputPath = "/home/ubuntu/datasets/ValidationDataset.csv";
            String modelPath = "/home/ubuntu/models/random-forest-model";
            String outputPath = "/home/ubuntu/predictions/predictions.txt";

            // Load the dataset
            Dataset<Row> validationData = spark.read()
                    .format("csv")
                    .option("header", "true")
                    .option("sep", ";")
                    .load(inputPath);

            validationData.printSchema();
            validationData.show(5);

            String[] cleanedColumns = Arrays.stream(validationData.columns())
                    .map(col -> col.replace("\"", "").trim())
                    .toArray(String[]::new);
            validationData = validationData.toDF(cleanedColumns);

            validationData.printSchema(); // Verify cleaned column names
            validationData.show(5);

            // Preprocess columns
            for (String colName : validationData.columns()) {
                if (!colName.equals("quality")) {
                    validationData = validationData.withColumn(colName, validationData.col(colName).cast("float"));
                }
            }
            validationData = validationData.withColumnRenamed("quality", "label");

            // Select feature columns
            String[] featureCols = {
                    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
                    "density", "pH", "sulphates", "alcohol"
            };

            // Assemble features
            VectorAssembler assembler = new VectorAssembler()
                    .setInputCols(featureCols)
                    .setOutputCol("features");

            Dataset<Row> assembledData = assembler.transform(validationData).select("features", "label");

            assembledData.printSchema();
            assembledData.show(5);

            // Convert DataFrame to JavaRDD<LabeledPoint>
            JavaRDD<LabeledPoint> labeledPoints = toLabeledPoint(sc, assembledData);

            // Load the pre-trained Random Forest model
            RandomForestModel model = RandomForestModel.load(sc.sc(), modelPath);

            System.out.println("Model loaded successfully");

            // Make predictions
            JavaRDD<Tuple2<Double, Double>> labelsAndPredictions = labeledPoints.map(lp ->
                    new Tuple2<>(lp.label(), model.predict(lp.features()))
            );

            // Save predictions to a file
            savePredictionsToFile(labelsAndPredictions, outputPath);

            // Evaluate the model
            MulticlassMetrics metrics = new MulticlassMetrics(labelsAndPredictions.rdd());
            System.out.println("Evaluation Metrics:");
            System.out.println("Weighted F1-score: " + metrics.weightedFMeasure());
            System.out.println("Confusion Matrix:\n" + metrics.confusionMatrix());
            System.out.println("Weighted Precision: " + metrics.weightedPrecision());
            System.out.println("Weighted Recall: " + metrics.weightedRecall());
            System.out.println("Accuracy: " + metrics.accuracy());

            // Calculate test error
            long testErrors = labelsAndPredictions.filter(lp -> !lp._1().equals(lp._2())).count();
            double testErrorRate = (double) testErrors / labeledPoints.count();
            System.out.println("Test Error Rate: " + testErrorRate);

        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            // Close the SparkContext
            sc.close();
        }
    }

    private static JavaRDD<LabeledPoint> toLabeledPoint(JavaSparkContext sc, Dataset<Row> df) {
        return df.toJavaRDD().map(row -> {
            double label = row.getDouble(row.fieldIndex("label"));
            org.apache.spark.ml.linalg.Vector mlVector = row.getAs("features");
            org.apache.spark.mllib.linalg.Vector mllibVector = org.apache.spark.mllib.linalg.Vectors.fromML(mlVector);
            return new LabeledPoint(label, mllibVector);
        });
    }

    private static void savePredictionsToFile(JavaRDD<Tuple2<Double, Double>> labelsAndPredictions, String outputPath) {
        try {
            // Collect all predictions to the driver
            List<String> formattedPredictions = labelsAndPredictions
                    .map(pair -> "Label: " + pair._1 + ", Prediction: " + pair._2)
                    .collect();

            // Write predictions to a single file
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputPath))) {
                for (String prediction : formattedPredictions) {
                    writer.write(prediction);
                    writer.newLine();
                }
            }

            System.out.println("Predictions saved to: " + outputPath);
        } catch (IOException e) {
            System.err.println("Error writing predictions to file: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
