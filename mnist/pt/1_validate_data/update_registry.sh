docker build -t pipeline-mnist-validate-data:1.0.0 .

docker tag pipeline-mnist-validate-data:1.0.0 kubeflow-registry.default.svc.cluster.local:30000/pipeline-mnist-validate-data:1.0.0

docker push kubeflow-registry.default.svc.cluster.local:30000/pipeline-mnist-validate-data:1.0.0
