package com.example;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import java.io.IOException;
import java.net.URI;
import java.util.List;
import java.util.ArrayList;
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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Tuple2;

public class WineQualityPrediction {
    private static final Logger logger = LoggerFactory.getLogger(WineQualityPrediction.class);

    public static void main(String[] args) {
        logger.info("Starting WineQualityPrediction...");

        // Configure Spark app
        SparkConf conf = new SparkConf()
            .setAppName("WineQualityPrediction")
            .setMaster("local[*]")
            .set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem");

        // Initialize Spark context and session
        JavaSparkContext sc = new JavaSparkContext(conf);
        SparkSession spark = SparkSession.builder().config(conf).getOrCreate();

        logger.info("Spark context initialized.");

        try {
            String inputPath = "s3a://winesparkbucket/ValidationDataset.csv";
            String modelPath = "s3a://winesparkbucket/random-forest-model";
            String outputPath = "s3a://winesparkbucket/predictions.txt";
            String evaluationsPath = "s3a://winesparkbucket/evaluations.txt";

            // Load the validation dataset from s3
            Dataset<Row> validationData = spark.read()
                    .format("csv")
                    .option("header", "true")
                    .option("sep", ";")
                    .load(inputPath);

            // Cleaning column names
            validationData = updateColumnNames(validationData);

            // Assembling features and the label column
            Dataset<Row> assembledData = assembleFeatures(validationData, "quality");

            // Convert data into LabeledPoint format for use with MLlib models
            JavaRDD<LabeledPoint> labeledPoints = toLabeledPoint(sc, assembledData);

            // Load the pre-trained Random Forest model from S3
            RandomForestModel model = RandomForestModel.load(sc.sc(), modelPath);

            // Generate predictions using the loaded model
            JavaRDD<Tuple2<Double, Double>> labelsAndPredictions = predictLabels(labeledPoints, model);

            // Save the predictions to an S3 file
            savePredictionsToFile(labelsAndPredictions, outputPath);

            // Evaluate model performance and save metrics to an S3 file
            evaluateAndSaveMetrics(labelsAndPredictions, evaluationsPath);

        } catch (Exception e) {
            logger.error("An error occurred:", e);
        } finally {
            sc.close();
        }
    }

    // Cleans and updates column names in the dataset.
    private static Dataset<Row> updateColumnNames(Dataset<Row> data) {
        String[] originalColumns = data.columns();
        String[] cleanedColumns = Arrays.stream(originalColumns)
                                        .map(col -> col.replaceAll("\"", "").trim())
                                        .toArray(String[]::new);
        return data.toDF(cleanedColumns);
    }

    // Assembles features from the dataset into a single column and prepares the label column.
    private static Dataset<Row> assembleFeatures(Dataset<Row> data, String labelColumn) {
        List<String> featureColumns = new ArrayList<>();
        for (String column : data.columns()) {
            if (!column.equals(labelColumn)) {
                featureColumns.add(column);
            }
        }
        data = data.withColumn("label", data.col(labelColumn).cast("double"))
                   .drop(labelColumn);

        for (String colName : featureColumns) {
            data = data.withColumn(colName, data.col(colName).cast("float"));
        }

        // Assemble feature columns into a single "features" column
        VectorAssembler assembler = new VectorAssembler()
            .setInputCols(featureColumns.toArray(new String[0]))
            .setOutputCol("features");

        return assembler.transform(data).select("features", "label");
    }

    // Converts a Dataset<Row> into an RDD of LabeledPoint for MLlib compatibility
    private static JavaRDD<LabeledPoint> toLabeledPoint(JavaSparkContext sc, Dataset<Row> data) {
        return data.toJavaRDD().map(row -> {
            double label = row.getDouble(row.fieldIndex("label"));
            org.apache.spark.ml.linalg.Vector features = row.getAs("features");
            org.apache.spark.mllib.linalg.Vector mllibFeatures = org.apache.spark.mllib.linalg.Vectors.fromML(features);
            return new LabeledPoint(label, mllibFeatures);
        });
    }

    // Predicts labels for the dataset using the provided Random Forest model.
    private static JavaRDD<Tuple2<Double, Double>> predictLabels(JavaRDD<LabeledPoint> data, RandomForestModel model) {
        return data.map(lp -> new Tuple2<>(lp.label(), model.predict(lp.features())));
    }

    // Saves predictions to an S3 file in the specified format.
    private static void savePredictionsToFile(JavaRDD<Tuple2<Double, Double>> predictions, String path) throws IOException {
        List<String> formattedPredictions = predictions.map(pair -> "Label: " + pair._1 + ", Prediction: " + pair._2).collect();
        writeToS3(path, formattedPredictions);
    }

    // Evaluates the model's predictions and saves evaluation metrics to an S3 file.
    private static void evaluateAndSaveMetrics(JavaRDD<Tuple2<Double, Double>> predictions, String path) throws IOException {
        MulticlassMetrics metrics = new MulticlassMetrics(predictions.rdd());
        List<String> results = new ArrayList<>();
        results.add("Evaluation Metrics:");
        results.add("Weighted F1-score: " + metrics.weightedFMeasure());
        results.add("Confusion Matrix:\n" + metrics.confusionMatrix());
        results.add("Weighted Precision: " + metrics.weightedPrecision());
        results.add("Weighted Recall: " + metrics.weightedRecall());
        results.add("Accuracy: " + metrics.accuracy());

        writeToS3(path, results);
    }

    // Writes a list of strings to an S3 file using Hadoop's FileSystem API.
    private static void writeToS3(String path, List<String> lines) throws IOException {
        Configuration hadoopConfig = new Configuration();
        FileSystem fs = FileSystem.get(URI.create(path), hadoopConfig);
        FSDataOutputStream outputStream = fs.create(new Path(path), true);
        for (String line : lines) {
            outputStream.writeBytes(line + "\n");
        }
        outputStream.close();
        fs.close();
    }
}