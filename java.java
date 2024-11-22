import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.*;

import static org.apache.spark.sql.functions.col;

public class WineQualityTrainer {
    public static void main(String[] args) {
        // Spark Session and Context setup
        SparkSession spark = SparkSession.builder()
                .appName("WineQualityTrainer")
                .master("local[*]")
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

        // Load training data
        Dataset<Row> trainingData = spark.read()
                .option("header", "true")
                .schema(schema)
                .csv("TrainingDataset.csv");

        // Load validation data
        Dataset<Row> validationData = spark.read()
                .option("header", "true")
                .schema(schema)
                .csv("ValidationDataset.csv");

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
        model.save("logistic-regression-model");

        spark.stop();
    }
}
