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

        // Configure Spark
        SparkConf conf = new SparkConf()
            .setAppName("WineQualityPrediction")
            .setMaster("local[*]")
            .set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem");

        JavaSparkContext sc = new JavaSparkContext(conf);
        SparkSession spark = SparkSession.builder().config(conf).getOrCreate();

        logger.info("Spark context initialized.");

        try {
            String inputPath = "s3a://winesparkbucket/ValidationDataset.csv";
            String modelPath = "s3a://winesparkbucket/random-forest-model";
            String outputPath = "s3a://winesparkbucket/predictions.txt";
            String evaluationsPath = "s3a://winesparkbucket/evaluations.txt";

            Dataset<Row> validationData = spark.read()
                    .format("csv")
                    .option("header", "true")
                    .option("sep", ";")
                    .load(inputPath);

            validationData = updateColumnNames(validationData);

            Dataset<Row> assembledData = assembleFeatures(validationData, "quality");
            JavaRDD<LabeledPoint> labeledPoints = toLabeledPoint(sc, assembledData);

            RandomForestModel model = RandomForestModel.load(sc.sc(), modelPath);
            JavaRDD<Tuple2<Double, Double>> labelsAndPredictions = predictLabels(labeledPoints, model);

            savePredictionsToFile(labelsAndPredictions, outputPath);

            evaluateAndSaveMetrics(labelsAndPredictions, evaluationsPath);

        } catch (Exception e) {
            logger.error("An error occurred:", e);
        } finally {
            sc.close();
        }
    }

    private static Dataset<Row> updateColumnNames(Dataset<Row> data) {
        String[] originalColumns = data.columns();
        String[] cleanedColumns = Arrays.stream(originalColumns)
                                        .map(col -> col.replaceAll("\"", "").trim())
                                        .toArray(String[]::new);
        return data.toDF(cleanedColumns);
    }

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

        VectorAssembler assembler = new VectorAssembler()
            .setInputCols(featureColumns.toArray(new String[0]))
            .setOutputCol("features");

        return assembler.transform(data).select("features", "label");
    }

    private static JavaRDD<LabeledPoint> toLabeledPoint(JavaSparkContext sc, Dataset<Row> data) {
        return data.toJavaRDD().map(row -> {
            double label = row.getDouble(row.fieldIndex("label"));
            org.apache.spark.ml.linalg.Vector features = row.getAs("features");
            org.apache.spark.mllib.linalg.Vector mllibFeatures = org.apache.spark.mllib.linalg.Vectors.fromML(features);
            return new LabeledPoint(label, mllibFeatures);
        });
    }

    private static JavaRDD<Tuple2<Double, Double>> predictLabels(JavaRDD<LabeledPoint> data, RandomForestModel model) {
        return data.map(lp -> new Tuple2<>(lp.label(), model.predict(lp.features())));
    }

    private static void savePredictionsToFile(JavaRDD<Tuple2<Double, Double>> predictions, String path) throws IOException {
        List<String> formattedPredictions = predictions.map(pair -> "Label: " + pair._1 + ", Prediction: " + pair._2).collect();
        writeToS3(path, formattedPredictions);
    }

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