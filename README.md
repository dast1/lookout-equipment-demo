# Amazon Lookout for Equipment Demo
Amazon Lookout for Equipment uses the data from your sensors to detect abnormal equipment behavior, so you can take action before machine failures occur and avoid unplanned downtime.

Amazon Lookout for Equipment analyzes the data from your sensors, such as pressure, flow rate, RPMs, temperature, and power to automatically train a specific ML model based on just your data, for your equipment – with no ML expertise required. Lookout for Equipment uses your unique ML model to analyze incoming sensor data in real-time and accurately identify early warning signs that could lead to machine failures. This means you can detect equipment abnormalities with speed and precision, quickly diagnose issues, take action to reduce expensive downtime, and reduce false alerts.

### Installation instructions
[Create an AWS account](https://portal.aws.amazon.com/gp/aws/developer/registration/index.html) if you do not already have one and login.

Navigate to the SageMaker console and create a new instance. Using an **ml.m5.xlarge instance** with a **5 GB attached EBS volume** is recommended to process the dataset comfortably. To enable exploration of big timeseries dataset, you might need to increase the EBS volume size. Some plots can take up a significant amount of memory: in such exploration, it's not unusual to move to bigger memory optimized instance (like the **ml.m5.4xlarge** one).

You need to ensure that this notebook instance has an IAM role which allows it to call the Amazon Lookout for Equipment APIs:
1. In your IAM console, look for the SageMaker execution role endorsed by your notebook instance (a role with a name like AmazonSageMaker-ExecutionRole-yyyymmddTHHMMSS)
2. === TO DO ===

Your SageMaker notebook instance can now call the Lookout for Equipment APIs.

You can know navigate back to the Amazon SageMaker console, then to the Notebook Instances menu. Start your instance and launch either Jupyter or JupyterLab session. From there, you can launch a new terminal and clone this repository into your local development machine using `git clone`.