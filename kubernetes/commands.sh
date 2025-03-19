kubectl create -f my-secret-eval.yml

kubectl create -f my-deployment-eval.yml

kubectl create -f my-service-eval.yml

kubectl create -f my-ingress-eval.yml

# Checking if everything works
# After deploying your YAML files, check the status of your pods:
kubectl get pods
# We see 3 pods running
# NAME                        READY   STATUS              RESTARTS   AGE
# user-api-68cdbd7849-8mnxv   0/2     ContainerCreating   0          10s
# user-api-68cdbd7849-b4gww   0/2     ContainerCreating   0          10s
# user-api-68cdbd7849-qxvdd   0/2     ContainerCreating   0          10s
# You should see three running pods (user-api), each containing two containers (mysql-database and fastapi). If any pods are in CrashLoopBackOff or Error state, describe them for details:

kubectl describe pod user-api-68cdbd7849-8mnxv
# Check API logs
kubectl logs user-api-68cdbd7849-8mnxv -c fastapi  
# Check database logs
kubectl logs user-api-68cdbd7849-8mnxv -c mysql-database  

# Check deployment and services
kubectl get deployments
kubectl get services

# Verify secrets
kubectl get secrets
kubectl get secret db -o yaml

# Test database connection insidde pod
# not found container fastapi
kubectl exec -it user-api-68cdbd7849-8mnxv -c fastapi -- /bin/sh

# Inside the container, test the database connection:
# mysql not found
mysql -h 127.0.0.1 -u root -p

# Test api inside cluster
kubectl port-forward svc/user-api-service 8000:8000

# Now, open another terminal and run:
curl http://127.0.0.1:8000/docs

# Test ingress
kubectl get ingress

# Test if external IP is assigned:
curl http://<ingress-ip>:8000/docs

# Check logs for errors
kubectl logs -l app=user-api -c fastapi
kubectl logs -l app=user-api -c mysql-database


# Delete everything
kubectl delete deployment --all
kubectl delete service --all
kubectl delete ingress --all
kubectl delete secret --all