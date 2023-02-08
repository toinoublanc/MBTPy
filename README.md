# **MBTPy**

**'MBT**I' Personnality-Type Indicator prediction model built with **Py**thon, FastAPI & Docker.

* **Input** : Text (ex: online posts)

- **Output** : 4-letters 'MBTI'  prediction.

The Myers-Briggs Type Indicator (MBTI) test is an introspective self personality assessment questionnaire that is based on the theories of Carl Jung.

It is widely used in the fields of psychology, education, and business as a tool for understanding personality and behavior.

The test consists of a series of questions that are used to determine an individual's psychological preferences along four axis :

1. Extraversion (E) vs. Introversion (I)
2. Sensing (S) vs. Intuition (N)
3. Thinking (T) vs. Feeling (F)
4. Judging (J) vs. Perceiving (P)

Based on their answers, individuals are assigned a four-letter personality type, such as INTJ (Introverted, Intuitive, Thinking, Judging) or ESFP (Extraverted, Sensing, Feeling, Perceiving).

It's important to note that the MBTI test is not a scientifically validated test and it has been criticized for lack of empirical evidence and reliability, but it is still widely used as a tool for personal development, self-awareness and, in the present case, training and deploying a machine learning model.

## Repository structure

```bash
MBTPy
├───api
│   ├───database
│   ├───processing
│   └───saved_models
├───data
│   └───raw
├───kubernetes
└───notebooks
    └───models
```

## Instructions to launch the API :

### Launch without container

1. fork this repo or download the code and unzip all files in a same directory.
2. Go in the `/api/` directory and make sure the environment matches packages and versions listed in `requirements.txt` (run `pip install -r requirements.txt` if needed).
3. Run `uvicorn main:app --reload`.
4. The API will be serving at http://localhost:8000

### Launch in a Docker container

1. Make sure you have `Docker` installed.
2. Pull the image from DockerHub :

```bash
docker pull toinoublanc/mbtpy-api:1.1.0
```

> Alternatively, if you want to build the image locally, go in the `/api/` directory and run
> `docker image build .`.

3. Run the following commands

```bash
docker run toinoublanc/mbtpy-api:1.1.0 -p 8000:8000 
```

4. The API will be serving at http://localhost:8000

### Deploy on a Kubernetes cluster (3 pods)

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

## Routers

### Routers related to user management

- GET  `/user/all` : Read all users from the table `user` of the database.
- POST  `/user/new` : Create a new user in the table `user` of the database.
- GET `/user/{user_id}` : Read a specific user from the table `user` of the database.
- DELETE `/user/{user_id}` : Delete a specific user from the table `user` of the database.
- PATCH `/user/{user_id}` : Update a specific user from the table `user` of the database.

### Routers related to prediction

- GET `/prediction/all` : Read all previous predictions stored in the table `prediction` of the database.
- POST  `/prediction/type` : Predict the 4-letters personnality type from a sample of text and store it in the table `prediction` of the database. Requires an API key (default value : `'apikey1234'`).

### Routers related to API testing

- GET `/home` : Basic endpoint.
- GET `/auth` : Says Hello to authentified users only.
- GET `/auth/admin` : Says Hello to admin users only.
- GET `/key` : Check API Key validity.
- POST `/user/demo` : Fill the `user` table of the database with 4 demo users (1 'admin' and 3 'standard'. Credentials of the 'admin' demo user are :  `'username' = 'Neo'` and `'password' = 'admin'`).
- GET `/test/coffee` : Purposely triggers error 418 to check if API is functional.

That's all folks !
