# Use an official OpenJDK runtime as the base image
FROM openjdk:11-jre-slim

# Install necessary dependencies (e.g., Spark, Hadoop, curl, etc.)
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    python3 \
    python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables for Spark and Hadoop
ENV SPARK_VERSION=3.5.3
ENV HADOOP_VERSION=3.3.1
ENV SPARK_HOME=/opt/spark
ENV HADOOP_HOME=/opt/hadoop
ENV PATH=$SPARK_HOME/bin:$HADOOP_HOME/bin:$PATH

# Download and install Spark
RUN wget https://archive.apache.org/dist/spark/spark-$SPARK_VERSION/spark-$SPARK_VERSION-bin-hadoop3.tgz && \
    tar -xzf spark-$SPARK_VERSION-bin-hadoop3.tgz -C /opt && \
    mv /opt/spark-$SPARK_VERSION-bin-hadoop3 $SPARK_HOME && \
    rm spark-$SPARK_VERSION-bin-hadoop3.tgz

# Download and install Hadoop
RUN wget https://archive.apache.org/dist/hadoop/common/hadoop-$HADOOP_VERSION/hadoop-$HADOOP_VERSION.tar.gz && \
    tar -xzf hadoop-$HADOOP_VERSION.tar.gz -C /opt && \
    mv /opt/hadoop-$HADOOP_VERSION $HADOOP_HOME && \
    rm hadoop-$HADOOP_VERSION.tar.gz

# Set Hadoop configuration for S3A
COPY core-site.xml /opt/hadoop/etc/hadoop/
COPY hdfs-site.xml /opt/hadoop/etc/hadoop/

# Install any required Python dependencies (if needed for future features)
RUN pip3 install numpy pandas

# Copy the application JAR into the container
COPY target/wine-quality-prediction-1.0-SNAPSHOT.jar /opt/spark-apps/

# Default command to run your application
CMD ["spark-submit", \
     "--class", "com.example.WineQualityPrediction", \
     "--master", "local[*]", \
     "/opt/spark-apps/wine-quality-prediction-1.0-SNAPSHOT.jar"]
