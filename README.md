# Spark Cluster Setup and Test

This README provides a step-by-step guide for setting up a Spark cluster on AWS EC2 instances, submitting a Spark job, and verifying results.

## Prerequisites
1. **AWS EC2 Instances**:
   - 1 Master Node
   - Multiple Worker Nodes
2. **Installed Software**:
   - Apache Spark (e.g., version 3.5.3)
   - Java (e.g., version 11)
3. **Security Groups**:
   - Allow communication between all cluster nodes on ports `7077` (Spark Master), `8080` (Master Web UI), `8081` (Worker Web UI), and ephemeral ports (e.g., `1024–65535`).
4. **SSH Access**:
   - Configure SSH to connect to each EC2 instance.

---

## Steps to Set Up and Test

### 1. Start Spark Services
#### Master Node:
Run the Spark master on the designated EC2 instance:
```bash
/opt/spark/sbin/start-master.sh
```
#### Verify the Master is running:
```cat /opt/spark/logs/spark-ubuntu-org.apache.spark.deploy.master.Master-*.out
```
#### Worker Nodes:
Run the Spark worker on each designated EC2 instance:

```
/opt/spark/sbin/start-worker.sh spark://<MASTER_IP>:7077
cat /opt/spark/logs/spark-ubuntu-org.apache.spark.deploy.worker.Worker-*.out
```

### 2. Verify Cluster Connectivity
Open the Master Web UI in a browser: http://<MASTER_IP>:8080.
Ensure all Worker nodes are listed under the "Workers" section.


