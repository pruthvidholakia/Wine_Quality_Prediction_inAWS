# Use Python 3.9 slim image as the base
FROM python:3.9-slim

# Install necessary packages and manually install OpenJDK 8
RUN apt-get update && apt-get install -y \
    wget tar unzip curl \
    && wget --no-verbose -O /tmp/openjdk.tar.gz \
       https://github.com/AdoptOpenJDK/openjdk8-binaries/releases/download/jdk8u292-b10/OpenJDK8U-jdk_x64_linux_hotspot_8u292b10.tar.gz \
    && mkdir -p /usr/lib/jvm \
    && tar -xf /tmp/openjdk.tar.gz -C /usr/lib/jvm \
    && rm /tmp/openjdk.tar.gz

# Set environment variables for Java and PySpark
ENV JAVA_HOME=/usr/lib/jvm/jdk8u292-b10
ENV PATH="$JAVA_HOME/bin:$PATH"
ENV PYSPARK_DRIVER_PYTHON=python3
ENV PYSPARK_PYTHON=python3

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Install Apache Spark
RUN wget --no-verbose -O apache-spark.tgz "https://archive.apache.org/dist/spark/spark-3.1.2/spark-3.1.2-bin-hadoop3.2.tgz" \
    && mkdir -p /opt/spark \
    && tar -xf apache-spark.tgz -C /opt/spark --strip-components=1 \
    && rm apache-spark.tgz

# Download Hadoop AWS dependencies
RUN wget https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.2.0/hadoop-aws-3.2.0.jar -P /opt/spark/jars/ \
    && wget https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.11.901/aws-java-sdk-bundle-1.11.901.jar -P /opt/spark/jars/

# Set Spark environment variables
ENV SPARK_HOME=/opt/spark
ENV PATH="$SPARK_HOME/bin:$PATH"

# Add core-site.xml configuration for S3
RUN echo '<configuration>' > /opt/spark/conf/core-site.xml \
    && echo '    <property>' >> /opt/spark/conf/core-site.xml \
    && echo '        <name>fs.s3a.impl</name>' >> /opt/spark/conf/core-site.xml \
    && echo '        <value>org.apache.hadoop.fs.s3a.S3AFileSystem</value>' >> /opt/spark/conf/core-site.xml \
    && echo '    </property>' >> /opt/spark/conf/core-site.xml \
    && echo '    <property>' >> /opt/spark/conf/core-site.xml \
    && echo '        <name>fs.s3a.aws.credentials.provider</name>' >> /opt/spark/conf/core-site.xml \
    && echo '        <value>com.amazonaws.auth.DefaultAWSCredentialsProviderChain</value>' >> /opt/spark/conf/core-site.xml \
    && echo '    </property>' >> /opt/spark/conf/core-site.xml \
    && echo '    <property>' >> /opt/spark/conf/core-site.xml \
    && echo '        <name>fs.s3a.endpoint</name>' >> /opt/spark/conf/core-site.xml \
    && echo '        <value>s3.amazonaws.com</value>' >> /opt/spark/conf/core-site.xml \
    && echo '    </property>' >> /opt/spark/conf/core-site.xml \
    && echo '</configuration>' >> /opt/spark/conf/core-site.xml

# Create directories
RUN mkdir -p /code/data/csv /code/data/model /code/src

# Copy local files into the container
COPY train_wineQuality.py /code/src/
COPY predict_wineQuality.py /code/src/
COPY TrainingDataset.csv /code/data/csv/
COPY ValidationDataset.csv /code/data/csv/
COPY optimized_wine_model_rf.model /code/data/model/

# Set working directory
WORKDIR /code

# Sequentially run training and testing scripts
ENTRYPOINT ["/bin/bash", "-c", "/opt/spark/bin/spark-submit /code/src/train_wineQuality.py && /opt/spark/bin/spark-submit /code/src/predict_wineQuality.py"]

