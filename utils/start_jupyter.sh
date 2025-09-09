TOKEN=`curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"`
INSTANCE_ID=`curl -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id`

cd /home/ec2-user
source /home/ec2-user/.bashrc
source /opt/intel/openvino/bin/setupvars.sh
/home/ec2-user/.local/bin/jupyter notebook --no-browser --NotebookApp.allow_password_change=False --NotebookApp.token="$INSTANCE_ID" --ip 0.0.0.0 --port 8888 > /tmp/jupyter.out 2>&1 &
