# TF-Metrics-Custom
A repo that contains custom TensorFlow implementations of machine learning metrics.

Estimator is a high-level api in TensorFLow, and it's high-efficent and useful when we build an Deep Learning Task. 

There is an inconvenience in estimator that you must use the method in tf.metric to evaluate your model, and it's necessary to use TensorFlow operator to define the metric formulation if you want to customize a new metric. There are some common metrics in the tf.metrics package which usually used in the tranditional classification or regression tasks. But for some task that need complex data postprocessing, we have to write the metric by oueselves.

It is well known that the TensorFlow operator api is too difficult to use. So, I will upload some metrics wrote by myself to this repo.

## The metrics list:
- Knowledge Graph Completion Task：
  - MRR(Mean Reciprocal Rank)
  - hit@N   



 
