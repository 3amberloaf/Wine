#!/bin/bash
# Run Spark job
"$SPARK_HOME/bin/spark-submit" \
  --class com.example.WineQualityPrediction \
  --master local[*] \
  /home/ubuntu/jars/wine-quality-prediction-1.0-SNAPSHOT.jar
