# Use an official OpenJDK runtime as the base image with JDK for Java development
FROM openjdk:11-jdk-slim

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    python3 \
    python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables for Spark and Hadoop
ENV SPARK_VERSION=3.5.3
ENV HADOOP_VERSION=3.3.6
ENV SPARK_HOME=/opt/spark
ENV HADOOP_HOME=/opt/hadoop
ENV PATH=$SPARK_HOME/bin:$PATH
ENV SPARK_CLASSPATH=$HADOOP_HOME/share/hadoop/tools/lib/hadoop-aws-${HADOOP_VERSION}.jar:$HADOOP_HOME/share/hadoop/tools/lib/aws-java-sdk-bundle-1.11.1026.jar

# Optional AWS credentials for accessing S3
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID:-default_key}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY:-default_secret}
ENV AWS_SESSION_TOKEN=${AWS_SESSION_TOKEN:-default_session}

# Download and install Spark
RUN wget https://dlcdn.apache.org/spark/spark-3.5.3/spark-3.5.3-bin-hadoop3.tgz && \
    tar -xzf spark-3.5.3-bin-hadoop3.tgz -C /opt && \
    mv /opt/spark-3.5.3-bin-hadoop3 $SPARK_HOME && \
    rm spark-3.5.3-bin-hadoop3.tgz

# Download and install Hadoop
RUN wget https://dlcdn.apache.org/hadoop/common/hadoop-3.3.6/hadoop-3.3.6-aarch64.tar.gz && \
    tar -xzf hadoop-3.3.6-aarch64.tar.gz -C /opt && \
    mv /opt/hadoop-3.3.6 $HADOOP_HOME && \
    rm hadoop-3.3.6-aarch64.tar.gz

# Add Hadoop AWS dependencies
RUN wget https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/${HADOOP_VERSION}/hadoop-aws-${HADOOP_VERSION}.jar -P $HADOOP_HOME/share/hadoop/tools/lib/ && \
    wget https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.11.1026/aws-java-sdk-bundle-1.11.1026.jar -P $HADOOP_HOME/share/hadoop/tools/lib/

# Copy the entrypoint script into the container
COPY entrypoint.sh /opt/

# Make the entrypoint script executable
RUN chmod +x /opt/entrypoint.sh

# Copy the application JAR into the container
COPY target/wine-quality-prediction-1.0-SNAPSHOT.jar /opt/spark-apps/

# Set the entrypoint script
ENTRYPOINT ["/opt/entrypoint.sh"]
