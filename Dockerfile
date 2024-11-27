FROM openjdk:11-jre-slim

# Install necessary dependencies (e.g., Spark, Python, etc.)
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    python3 \
    python3-pip

# Set environment variables for Spark
ENV SPARK_VERSION=3.5.3
ENV HADOOP_VERSION=3.2
ENV SPARK_HOME=/opt/spark
ENV PATH=$SPARK_HOME/bin:$PATH

# Download and install Spark
RUN wget https://archive.apache.org/dist/spark/spark-$SPARK_VERSION/spark-$SPARK_VERSION-bin-hadoop$HADOOP_VERSION.tgz
RUN tar -xzf spark-$SPARK_VERSION-bin-hadoop$HADOOP_VERSION.tgz -C /opt
RUN rm spark-$SPARK_VERSION-bin-hadoop$HADOOP_VERSION.tgz

# Install any other dependencies, like Python packages for prediction
RUN pip3 install pandas numpy

# Copy your prediction code (JAR or Python script)
COPY prediction-app.jar /opt/spark-apps/

# Command to run the prediction
CMD ["spark-submit", "--class", "com.example.WineQualityPrediction", "/opt/spark-apps/prediction-app.jar"]
