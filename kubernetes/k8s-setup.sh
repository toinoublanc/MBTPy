###############################################################################
# KUBERNETES

#------------------------------------------------------------------------------
# DESIGN (self)

# Pull required images (Optional, that will be done automatically)
docker pull toinoublanc/mbtpy-mysql:latest
docker pull toinoublanc/mbtpy-api:latest
# Attention, éviter _ dans le nom des images, préférer - (sinon kubectl fait la tête)

mkdir kubernetes
cd kubernetes

nano mbtpy-deployment.yml
nano mbtpy-service.yml
nano mbtpy-ingress.yml

# Create Secret
# Creation secret via CLI
kubectl create secret generic mbtpy-secret --from literal password=password

# Création secret via configuration file
echo -n 'password' | base64 # Convert strings to base64
nano mbtpy-secret.yml # Create the manifest, using the base64 value



#------------------------------------------------------------------------------
# TO DEPLOY ON 3 PODS

# (if needed) Install minikube
# curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
# sudo install minikube-linux-amd64 /usr/local/bin/minikube

# (if needed) Install kubectl to interact with Kubernetes
# curl -LO https://storage.googleapis.com/kubernetes-release/release/v1.21.0/bin/linux/amd64/kubectl
# chmod +x ./kubectl
# sudo mv ./kubectl /usr/local/bin/kubectl
# kubectl version --client

# Launch minikube cluster
minikube start
minikube dashboard --url=true

# (if needed) Enable ingress
minikube addons enable ingress # Disabled by default

# (if launched from a distant machine) Allow dashboard to be reached from local machine
# kubectl proxy --address='0.0.0.0' --disable-filter=true

# Go to dashboard 
# Click on link, or use URL; it should look like these (ADAPT URL)
# from a distant machine : http://79.125.27.171:8001/api/v1/namespaces/kubernetes-dashboard/services/http:kubernetes-dashboard:/proxy/
# from local machine : http://127.0.0.1:62218/api/v1/namespaces/kubernetes-dashboard/services/http:kubernetes-dashboard:/proxy/

# (if needed) Move to the folder with kubernetes files
cd ./kubernetes/

# Apply Secret
kubectl apply -f mbtpy-secret.yml # from configuration file

# Create Deployment to launch pods
kubectl create -f mbtpy-deployment.yml # from configuration file

# Create Service to be able to access launched pods within the cluster
kubectl create -f mbtpy-service.yml # from configuration file

# Create Ingress to expose the Service 
kubectl create -f mbtpy-ingress.yml # from configuration file

# Wait until all workloads status is 'running' (green)

# Open a Shell terminal in a pod and check its status
apt update
apt install curl -y
curl -X GET -i http://localhost:8000/status


#------------------------------------------------------------------------------
# TO DELETE AND STOP

# minikube addons disable ingress
kubectl delete ingress mbtpy-ingress
kubectl delete service mbtpy-service
kubectl delete deployment mbtpy-deployment
kubectl delete secret mbtpy-secret
minikube stop


###############################################################################