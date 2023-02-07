# **MBTPy**

Personnality-Type Indicator prediction model - MLOps project

API build with FastAPI.

- Input : Text.
- Output : 4-letters 'MBTI' type prediction.

## Instructions to launch the API :

#### Lanch without container

1. fork this repo or download the code and unzip all files in a same directory.
2. Go in the `/api/` directory and make sure the environment matches packages and versions listed in `requirements.txt` (run `pip install -r requirements.txt` if needed).
3. Run `uvicorn main:app --reload`.
4. The API will be serving at http://localhost:8000

#### Launch in a Docker container

1. Make sure you have `Docker` installed.
2. Pull the image from DockerHub :

```bash
docker pull toinoublanc/mbtpy-api:1.0.0
```

> Alternatively, if you want to build the image locally, go in the `/api/` directory and run
> `docker image build .`.

3. Run the following commands

```bash
docker run toinoublanc/mbtpy-api:1.0.0 -p 8000:8000 
```

4. The API will be serving at http://localhost:8000

#### Deploy on a Kubernetes cluster (3 pods)

1. Make sure you have `Kubernetes`, `minikube` and `kubectl` installed.
2. Run the following commands to launch a minikube cluster

```bash
minikube start
minikube dashboard --url=true
```

> You might have to enable ingress. This feature is disabled by default for safety, so use it with caution.
>
> ```bash
> minikube addons enable ingress
> ```

3. If you launched the cluster on a distant machine, run the following command to allow the dashboard to be reached from local machine:

```bash
kubectl proxy --address='0.0.0.0' --disable-filter=true
```

4. Go to dashboard using the adress that should appears.
5. Go in the `/kubernetes/` folder and create a `Deployment` to launch the pods from the configuation file:

```bash
kubectl create -f mbtpy-deployment.yml
```

6. Now create a `Service` to make the launched pods reachable within the cluster

```bash
kubectl create -f mbtpy-service.yml
```

7. Then, create an `Ingress` to expose the `Service` to the outside.

```bash
kubectl create -f mbtpy-ingress.yml
```

8. Wait until all workloads status are 'running' (green). There you go. 3 pods.

> You can delete the previously created entities with :
>
> ```bash
> kubectl delete ingress mbtpy-ingress
> kubectl delete service mbtpy-service
> kubectl delete deployment mbtpy-deployment
> ```

Do not forget to stop the cluster when you're done:

```bash
minikube stop
```

That's all folks !
