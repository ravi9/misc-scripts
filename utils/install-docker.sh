# Install Docker, restart terminal after installing

curl -fsSL https://get.docker.com -o get-docker.sh &&
sudo sh get-docker.sh && 
sudo usermod -aG docker $USER
