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

// Train and validate wine quality dataset using a Random Forest model
public class WineQualityTraining {

    public static void main(String[] args) {
        // Configures Spark application single-node testing
        SparkConf conf = new SparkConf().setAppName("WineQualityTrainer").setMaster("spark://172.31.31.175:7077");
        JavaSparkContext sc = new JavaSparkContext(conf);
        SparkSession spark = SparkSession.builder().config(conf).getOrCreate();
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

        for (String colName : val.columns()) {
            if (!colName.equals("quality")) {
                val = val.withColumn(colName, val.col(colName).cast("float"));
            }
        }
        
        // Rename the "quality" column to "label"  for MLlib supervised learning models
        val = val.withColumnRenamed("quality", "label");

        // Combine feature columns into a single vector column ("features")
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(val.columns()) // Input columns are all columns except the label
                .setOutputCol("features"); // Output column is named "features"
        
        // Transform the data by applying the VectorAssembler and select only "features" and "label"
        Dataset<Row> df_tr = assembler.transform(val).select("features", "label");
        
        df_tr.show();

        // Convert the DataFrame into an RDD of LabeledPoint objects
        JavaRDD<LabeledPoint> dataset = toLabeledPoint(sc, df_tr);

        // Load a pre-trained Random Forest model from S3
        // RandomForestModel RFModel = RandomForestModel.load(sc.sc(), "/sparkwinebucket/trainingmodel.model/");
        RandomForestModel RFModel = RandomForestModel.load(sc.sc(), "/home/ubuntu/models/random-forest-model");

        // Confirm that the model was successfully loaded
        System.out.println("Model loaded successfully");

        // Use the Random Forest model to make predictions on the dataset
        JavaRDD<Double> predictions = RFModel.predict(dataset.map(LabeledPoint::features));

        // Map the dataset to an RDD for evaluation
        JavaRDD<Tuple2<Double, Double>> labelsAndPredictions = dataset
                .map(lp -> new Tuple2<>(lp.label(), RFModel.predict(lp.features())));

        Dataset<Row> labelPred = spark.createDataFrame(labelsAndPredictions, Tuple2.class).toDF("label", "Prediction");
        labelPred.show();

        MulticlassMetrics metrics = new MulticlassMetrics(labelPred);

        // Calculate and print the F1-score, a measure of model accuracy
        // Compute F1-score for all classes (weighted)
        double f1Score = metrics.weightedFMeasure();
        System.out.println("Weighted F1-score: " + f1Score);

        System.out.println(metrics.confusionMatrix());
        System.out.println(metrics.weightedPrecision());
        System.out.println(metrics.weightedRecall());
        System.out.println("Accuracy: " + metrics.accuracy());


        // Close the SparkContext
        sc.close();
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
