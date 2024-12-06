#!/bin/bash
/home/ubuntu/spark/bin/spark-submit \
    --class com.example.WineQualityPrediction \
    --master local[*] \
    /home/ubuntu/jars/wine-quality-prediction-1.0-SNAPSHOT-jar-with-dependencies.jar
