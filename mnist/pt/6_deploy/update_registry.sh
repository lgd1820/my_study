docker build -t pipeline-mnist-deploy:1.0.0 .

docker tag pipeline-mnist-deploy:1.0.0 kubeflow-registry.default.svc.cluster.local:30000/pipeline-mnist-deploy:1.0.0

docker push kubeflow-registry.default.svc.cluster.local:30000/pipeline-mnist-deploy:1.0.0
