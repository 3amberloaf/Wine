# Use an official OpenJDK runtime as the base image
FROM --platform=linux/arm64 openjdk:11-jdk-slim


# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    python3 \
    python3-pip \
    procps \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables for Spark and Hadoop
ENV SPARK_VERSION=3.5.3
ENV HADOOP_VERSION=3.3.6
ENV SPARK_HOME=/home/ubuntu/spark
ENV HADOOP_HOME=/home/ubuntu/hadoop
ENV PATH=$SPARK_HOME/bin:$PATH
ENV SPARK_CLASSPATH=$HADOOP_HOME/share/hadoop/tools/lib/hadoop-aws-${HADOOP_VERSION}.jar:$HADOOP_HOME/share/hadoop/tools/lib/aws-java-sdk-bundle-1.11.1026.jar

# Optional AWS credentials for accessing S3
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID:-default_key}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY:-default_secret}
ENV AWS_SESSION_TOKEN=${AWS_SESSION_TOKEN:-default_session}

# Download and install Spark
RUN mkdir -p /home/ubuntu/spark && \
    wget https://dlcdn.apache.org/spark/spark-3.5.3/spark-3.5.3-bin-hadoop3.tgz && \
    tar -xzf spark-3.5.3-bin-hadoop3.tgz -C /home/ubuntu/spark --strip-components=1 && \
    rm spark-3.5.3-bin-hadoop3.tgz

# Download and install Hadoop
RUN wget https://dlcdn.apache.org/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}-aarch64.tar.gz && \
    tar -xzf hadoop-${HADOOP_VERSION}-aarch64.tar.gz -C /opt && \
    mv /opt/hadoop-${HADOOP_VERSION} $HADOOP_HOME && \
    rm hadoop-${HADOOP_VERSION}-aarch64.tar.gz

# Add Hadoop AWS dependencies
RUN wget https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/${HADOOP_VERSION}/hadoop-aws-${HADOOP_VERSION}.jar -P $HADOOP_HOME/share/hadoop/tools/lib/ && \
    wget https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.11.1026/aws-java-sdk-bundle-1.11.1026.jar -P $HADOOP_HOME/share/hadoop/tools/lib/

# Copy the entrypoint script into the container
COPY entrypoint.sh /home/ubuntu/
RUN chmod +x /home/ubuntu/entrypoint.sh

# Copy the application JAR into the container
COPY target/wine-quality-prediction-1.0-SNAPSHOT-jar-with-dependencies.jar /home/ubuntu/jars/

# Set the entrypoint script
ENTRYPOINT ["/home/ubuntu/entrypoint.sh"]
