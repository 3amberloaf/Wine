# Use an official OpenJDK runtime as the base image
FROM openjdk:11-jre-slim

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    python3 \
    python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables for Spark and Hadoop
ENV SPARK_VERSION=3.5.3
ENV HADOOP_VERSION=3.3.1
ENV SPARK_HOME=/home/ubuntu/spark
ENV HADOOP_HOME=/opt/hadoop
ENV PATH=$SPARK_HOME/bin:$HADOOP_HOME/bin:$PATH
ENV SPARK_CLASSPATH=$HADOOP_HOME/share/hadoop/tools/lib/hadoop-aws-3.3.1.jar:$HADOOP_HOME/share/hadoop/tools/lib/aws-java-sdk-bundle-1.11.1026.jar

# Download and install Spark
RUN wget https://archive.apache.org/dist/spark/spark-$SPARK_VERSION/spark-$SPARK_VERSION-bin-hadoop3.tgz && \
    tar -xzf spark-$SPARK_VERSION-bin-hadoop3.tgz -C /home/ubuntu && \
    mv /home/ubuntu/spark-$SPARK_VERSION-bin-hadoop3 $SPARK_HOME && \
    rm spark-$SPARK_VERSION-bin-hadoop3.tgz

# Download and install Hadoop
RUN wget https://archive.apache.org/dist/hadoop/common/hadoop-$HADOOP_VERSION/hadoop-$HADOOP_VERSION.tar.gz && \
    tar -xzf hadoop-$HADOOP_VERSION.tar.gz -C /opt && \
    mv /opt/hadoop-$HADOOP_VERSION $HADOOP_HOME && \
    rm hadoop-$HADOOP_VERSION.tar.gz

# Add Hadoop AWS dependencies
RUN wget https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.1/hadoop-aws-3.3.1.jar -P $HADOOP_HOME/share/hadoop/tools/lib/ && \
    wget https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.11.1026/aws-java-sdk-bundle-1.11.1026.jar -P $HADOOP_HOME/share/hadoop/tools/lib/

# Copy the entrypoint script
COPY entrypoint.sh /opt/

# Make the entrypoint script executable
RUN chmod +x /opt/entrypoint.sh

# Copy the application JAR into the container
COPY target/wine-quality-prediction-1.0-SNAPSHOT.jar /home/ubuntu/jars/

# Set the entrypoint script
ENTRYPOINT ["/opt/entrypoint.sh"]