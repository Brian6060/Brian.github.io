#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layers/weighted_softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void WeightedSoftmaxWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void WeightedSoftmaxWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2);
  CHECK_EQ(bottom[1]->num(), bottom[0]->num()) << "Label num must match score num.";
  CHECK_EQ(bottom[2]->num(), bottom[0]->num()) << "Weight num must match score num.";

  CHECK_EQ(bottom[1]->channels(), 1) << "Label must have 1 channel.";
  CHECK_EQ(bottom[2]->channels(), 1) << "Weight must have 1 channel.";

  CHECK_EQ(bottom[1]->height(), bottom[0]->height()) << "Label height mismatch.";
  CHECK_EQ(bottom[1]->width(), bottom[0]->width()) << "Label width mismatch.";

  CHECK_EQ(bottom[2]->height(), bottom[0]->height()) << "Weight height mismatch.";
  CHECK_EQ(bottom[2]->width(), bottom[0]->width()) << "Weight width mismatch.";

  prob_.ReshapeLike(*bottom[0]);

  outer_num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  inner_num_ = bottom[0]->count() / (outer_num_ * channels_);

  vector<int> loss_shape(0);
  top[0]->Reshape(loss_shape);
}

template <typename Dtype>
void WeightedSoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* weight = bottom[2]->cpu_data();
  Dtype* prob_data = prob_.mutable_cpu_data();

  for (int n = 0; n < outer_num_; ++n) {
    for (int s = 0; s < inner_num_; ++s) {
      Dtype max_val = bottom_data[(n * channels_) * inner_num_ + s];
      for (int c = 1; c < channels_; ++c) {
        const int idx = (n * channels_ + c) * inner_num_ + s;
        max_val = std::max(max_val, bottom_data[idx]);
      }

      Dtype sum_exp = 0;
      for (int c = 0; c < channels_; ++c) {
        const int idx = (n * channels_ + c) * inner_num_ + s;
        prob_data[idx] = std::exp(bottom_data[idx] - max_val);
        sum_exp += prob_data[idx];
      }

      for (int c = 0; c < channels_; ++c) {
        const int idx = (n * channels_ + c) * inner_num_ + s;
        prob_data[idx] /= sum_exp;
      }
    }
  }

  Dtype loss = 0;
  const int count = outer_num_ * inner_num_;
  for (int n = 0; n < outer_num_; ++n) {
    for (int s = 0; s < inner_num_; ++s) {
      const int label_value = static_cast<int>(label[n * inner_num_ + s]);
      CHECK_GE(label_value, 0);
      CHECK_LT(label_value, channels_);

      const Dtype w = weight[n * inner_num_ + s];
      const int idx = (n * channels_ + label_value) * inner_num_ + s;
      const Dtype p = std::max(prob_data[idx], Dtype(1e-20));
      loss -= w * std::log(p);
    }
  }

  top[0]->mutable_cpu_data()[0] = loss / count;
}

template <typename Dtype>
void WeightedSoftmaxWithLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[2]) {
    LOG(FATAL) << this->type() << " Layer cannot backpropagate to weight inputs.";
  }
  if (!propagate_down[0]) {
    return;
  }

  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* weight = bottom[2]->cpu_data();
  const Dtype* prob_data = prob_.cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  caffe_copy(prob_.count(), prob_data, bottom_diff);

  const int count = outer_num_ * inner_num_;
  for (int n = 0; n < outer_num_; ++n) {
    for (int s = 0; s < inner_num_; ++s) {
      const int label_value = static_cast<int>(label[n * inner_num_ + s]);
      CHECK_GE(label_value, 0);
      CHECK_LT(label_value, channels_);

      const Dtype w = weight[n * inner_num_ + s];

      for (int c = 0; c < channels_; ++c) {
        const int idx = (n * channels_ + c) * inner_num_ + s;
        if (c == label_value) {
          bottom_diff[idx] = w * (bottom_diff[idx] - 1);
        } else {
          bottom_diff[idx] = w * bottom_diff[idx];
        }
      }
    }
  }

  const Dtype loss_weight = top[0]->cpu_diff()[0] / count;
  caffe_scal(prob_.count(), loss_weight, bottom_diff);
}

#ifdef CPU_ONLY
STUB_GPU(WeightedSoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(WeightedSoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(WeightedSoftmaxWithLoss);

}  // namespace caffe
