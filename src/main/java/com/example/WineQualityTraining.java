package com.example;

import scala.Tuple2;

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

import java.util.Arrays;

public class WineQualityTraining {

    public static void main(String[] args) {
        // Configures Spark application single-node testing
        SparkConf conf = new SparkConf().setAppName("WineQualityTrainer").setMaster("spark://172.31.31.175:7077");
        JavaSparkContext sc = new JavaSparkContext(conf);
        SparkSession spark = SparkSession.builder().config(conf).getOrCreate();

        try {
            String path = args[0];

            // Load the validation dataset from the given path in CSV format
            Dataset<Row> val = spark.read()
                    .format("csv")
                    .option("header", "true")
                    .option("sep", ";")
                    .load(path);

            // Prints and displays loaded data
            val.printSchema();
            val.show();

            // Cast feature columns to float and rename 'quality' to 'label'
            for (String colName : val.columns()) {
                if (!colName.equals("quality")) {
                    val = val.withColumn(colName, val.col(colName).cast("float"));
                }
            }
            val = val.withColumn("label", val.col("quality").cast("double")).drop("quality");

            // Dynamically extract feature columns
            String[] featureCols = Arrays.stream(val.columns())
                    .filter(col -> !col.equals("label"))
                    .toArray(String[]::new);

            // Combine feature columns into a single vector column ("features")
            VectorAssembler assembler = new VectorAssembler()
                    .setInputCols(featureCols)
                    .setOutputCol("features");

            Dataset<Row> df_tr = assembler.transform(val).select("features", "label");

            df_tr.printSchema();
            df_tr.show();

            // Validate schema
            if (!Arrays.asList(df_tr.columns()).contains("label")) {
                throw new IllegalArgumentException("Label column is missing after preprocessing.");
            }

            // Convert the DataFrame into an RDD of LabeledPoint objects
            JavaRDD<LabeledPoint> dataset = toLabeledPoint(sc, df_tr);

            // Load the pre-trained Random Forest model
            RandomForestModel RFModel = RandomForestModel.load(sc.sc(), "/home/ubuntu/models/random-forest-model");

            // Confirm model was loaded
            System.out.println("Model loaded successfully");

            // Use the Random Forest model to make predictions on the dataset
            JavaRDD<Double> predictions = RFModel.predict(dataset.map(LabeledPoint::features));

            // Print the predictions
            predictions.take(10).forEach(prediction -> System.out.println("Prediction: " + prediction));

            // Map the dataset to an RDD for evaluation
            JavaRDD<Tuple2<Double, Double>> labelsAndPredictions = dataset
                    .map(lp -> new Tuple2<>(lp.label(), RFModel.predict(lp.features())));

            // Print the labels and predictions for a sample
            labelsAndPredictions.take(10).forEach(pair -> 
                System.out.println("Label: " + pair._1 + ", Prediction: " + pair._2)
            );

            // Create a DataFrame for displaying results
            Dataset<Row> labelPred = spark.createDataFrame(labelsAndPredictions, Tuple2.class).toDF("label", "Prediction");
            labelPred.show();

            // Evaluate the model using MulticlassMetrics
            JavaRDD<Tuple2<Object, Object>> metricsRDD = labelsAndPredictions.map(
                t -> new Tuple2<>((Object) t._1, (Object) t._2)
            );
            MulticlassMetrics metrics = new MulticlassMetrics(metricsRDD.rdd());

            // Calculate and print metrics
            System.out.println("Confusion Matrix:\n" + metrics.confusionMatrix());
            System.out.println("Weighted Precision: " + metrics.weightedPrecision());
            System.out.println("Weighted Recall: " + metrics.weightedRecall());
            System.out.println("Weighted F1-score: " + metrics.weightedFMeasure());
            System.out.println("Accuracy: " + metrics.accuracy());

        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            // Close the SparkContext
            sc.close();
        }
    }

    // Helper method 
    private static JavaRDD<LabeledPoint> toLabeledPoint(JavaSparkContext sc, Dataset<Row> df) {
        return df.toJavaRDD().map(row -> {
            double label = row.getDouble(row.fieldIndex("label")); // Extract the label
            // Convert ML Vector to MLlib Vector
            org.apache.spark.ml.linalg.Vector mlVector = row.getAs("features"); // Features column is ML Vector
            org.apache.spark.mllib.linalg.Vector mllibVector = org.apache.spark.mllib.linalg.Vectors.fromML(mlVector); // Convert to MLlib vector
            
            return new LabeledPoint(label, mllibVector); // Create LabeledPoint
        });
    }
}
