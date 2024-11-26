package com.example;
import java.io.IOException;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class WineQualityTrainer {
    public static void main(String[] args) {
        // Spark Session and Context setup
        SparkSession spark = SparkSession.builder()
                .appName("WineQualityTrainer")
                .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com")
                .config("spark.hadoop.fs.s3a.access.key", "ASIASKQKXK7AG4K7CJGR") 
                .config("spark.hadoop.fs.s3a.secret.key", "lhQrL2HmxfeTX3XxemR6Rvm4o7wThxlGhOsSQiB+") 
                .config("spark.hadoop.fs.s3a.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
                .getOrCreate();


        JavaSparkContext sc = new JavaSparkContext(spark.sparkContext());
        sc.setLogLevel("ERROR");

        // Schema definition
        StructType schema = new StructType(new StructField[]{
                new StructField("fixed_acidity", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("volatile_acidity", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("citric_acid", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("residual_sugar", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("chlorides", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("free_sulfur_dioxide", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("total_sulfur_dioxide", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("density", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("pH", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("sulphates", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("alcohol", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("quality", DataTypes.IntegerType, false, Metadata.empty())
        });

        // Load training data from S3 bucket
        Dataset<Row> trainingData = spark.read()
                .option("header", "true")
                .schema(schema)
                .csv("s3a://winesparkbucket/TrainingDataset.csv");

        // Load validation data from S3 bucket
        Dataset<Row> validationData = spark.read()
                .option("header", "true")
                .schema(schema)
                .csv("s3a://winesparkbucket/ValidationDataset.csv");

        // Assemble feature columns into a single vector column
        String[] featureCols = new String[]{
                "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
                "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
                "pH", "sulphates", "alcohol"
        };

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(featureCols)
                .setOutputCol("features");

        Dataset<Row> assembledTrainingData = assembler.transform(trainingData)
                .select("features", "quality")
                .withColumnRenamed("quality", "label");

        Dataset<Row> assembledValidationData = assembler.transform(validationData)
                .select("features", "quality")
                .withColumnRenamed("quality", "label");

        // Train logistic regression model
        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(10)
                .setRegParam(0.3)
                .setElasticNetParam(0.8);

        Dataset<Row> modelTrainingData = assembledTrainingData.select("features", "label");
        org.apache.spark.ml.classification.LogisticRegressionModel model = lr.fit(modelTrainingData);

        // Evaluate model on validation data
        Dataset<Row> predictions = model.transform(assembledValidationData);

        BinaryClassificationEvaluator evaluator = new BinaryClassificationEvaluator()
                .setLabelCol("label")
                .setRawPredictionCol("rawPrediction")
                .setMetricName("areaUnderROC");

        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Validation Accuracy: " + accuracy);

        // Save the model for future predictions
        try {
            model.save("logistic-regression-model");
            model.save("hdfs://path/to/save/model");

        } catch (IOException e) {
            e.printStackTrace();
        }

        spark.stop();
    }
}

