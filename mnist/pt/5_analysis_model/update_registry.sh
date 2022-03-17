docker build -t pipeline-mnist-analysis-model:1.0.0 .

docker tag pipeline-mnist-analysis-model:1.0.0 kubeflow-registry.default.svc.cluster.local:30000/pipeline-mnist-analysis-model:1.0.0

docker push kubeflow-registry.default.svc.cluster.local:30000/pipeline-mnist-analysis-model:1.0.0
