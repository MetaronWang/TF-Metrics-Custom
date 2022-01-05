import numpy as np
import tensorflow as tf


def mrr_metric(labels, predictions, weights=None, metrics_collections=None, updates_collections=None, name=None):
    '''
    A tf.metrics implementation of Mean Reciprocal Rank (MRR)

    :param labels: The ground truth labels of the data. A binary tensor with at least 2-dim that
        contain the label of each query(h,r,?) or (?,r,y)
    :param predictions: The prediction score tensor of the data, have the same shape with labels
    :param weights: Optional `Tensor` whose rank is either 0, or the same rank as
        `values`, and must be broadcastable to `values` (i.e., all dimensions must
        be either `1`, or the same as the corresponding `values` dimension).
    :param metrics_collections:An optional list of collections that `mean`
        should be added to.
    :param updates_collections:An optional list of collections that `update_op`
        should be added to.
    :param name: An optional variable_scope name.
    :return mrr:  A `Tensor` representing the current Mean Reciprocal Rank
    :return update_mrr_op: An operation that update the current Mean Reciprocal Rank.
    '''
    with tf.name_scope(name, 'mrr_metric', [predictions, labels, weights]) as scope:
        c1 = tf.argsort(predictions, axis=-1, direction='DESCENDING')
        c2 = tf.argsort(c1, axis=-1)
        true_place = tf.where(tf.equal(labels, 1))
        r = tf.gather_nd(c2 + 1, true_place)
        rr = 1 / r
        m_rr, update_mrr_op = tf.metrics.mean(rr)

        if metrics_collections:
            tf.add_to_collection(metrics_collections, m_rr)

        if updates_collections:
            tf.add_to_collections(updates_collections, update_mrr_op)

        return m_rr, update_mrr_op


def hitk_metric(labels, predictions, k, weights=None, metrics_collections=None, updates_collections=None, name=None):
    '''
        A tf.metrics implementation of hit rate in top-k (hit@k)

        :param labels: The ground truth labels of the data. A binary tensor with at least 2-dim that
            contain the label of each query(h,r,?) or (?,r,y),
        :param predictions: The prediction score tensor of the data, have the same shape with labels
        :param k: The top-k entry will be check hit or not
        :param weights: Optional `Tensor` whose rank is either 0, or the same rank as
            `values`, and must be broadcastable to `values` (i.e., all dimensions must
            be either `1`, or the same as the corresponding `values` dimension).
        :param metrics_collections:An optional list of collections that `mean`
            should be added to.
        :param updates_collections:An optional list of collections that `update_op`
            should be added to.
        :param name: An optional variable_scope name.
        :return mrr:  A `Tensor` representing the current Mean Reciprocal Rank
        :return update_mrr_op: An operation that update the current Mean Reciprocal Rank.
        '''

    with tf.name_scope(name, 'hitk_metric', [predictions, labels, weights]) as scope:
        c1 = tf.argsort(predictions, axis=-1, direction='DESCENDING')
        c2 = tf.argsort(c1, axis=-1)
        hit = tf.math.maximum(k - c2, 0) * labels
        hit_per_query = tf.math.minimum(1, tf.reduce_sum(hit, axis=-1))
        hit_percent = tf.reduce_sum(hit_per_query, axis=-1) / hit_per_query.get_shape().as_list()[0]

        m_rr, update_mrr_op = tf.metrics.mean(hit_percent)

        if metrics_collections:
            tf.add_to_collection(metrics_collections, m_rr)

        if updates_collections:
            tf.add_to_collections(updates_collections, update_mrr_op)

        return m_rr, update_mrr_op


if __name__ == '__main__':
    predictions = np.array([[5, 4, 3, 2, 1], [1, 2, 3, 4, 5], [2.3, 4, 2, 5, 1.2]], dtype=np.float)
    labels = np.array([[1, 0, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0]])
    value, op = mrr_metric(labels, predictions)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(op)
        print(value.eval())
