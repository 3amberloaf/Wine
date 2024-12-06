#!/bin/bash

/home/ubuntu/spark/bin/spark-submit \
    --class com.example.WineQualityPrediction \
    --master local[*] \
    --conf spark.hadoop.fs.s3a.access.key=${AWS_ACCESS_KEY_ID} \
    --conf spark.hadoop.fs.s3a.secret.key=${AWS_SECRET_ACCESS_KEY} \
    --conf spark.hadoop.fs.s3a.session.token=${AWS_SESSION_TOKEN} \
    --conf spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem \
    /home/ubuntu/jars/wine-quality-prediction-1.0-SNAPSHOT-jar-with-dependencies.jar
