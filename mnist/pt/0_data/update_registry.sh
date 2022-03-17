docker build -t pipeline-mnist-pre-data:1.0.0 .

docker tag pipeline-mnist-pre-data:1.0.0 kubeflow-registry.default.svc.cluster.local:30000/pipeline-mnist-pre-data:1.0.0

docker push kubeflow-registry.default.svc.cluster.local:30000/pipeline-mnist-pre-data:1.0.0
