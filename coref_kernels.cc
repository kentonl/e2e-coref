#include <map>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("ExtractSpans")
.Input("span_scores: float32")
.Input("candidate_starts: int32")
.Input("candidate_ends: int32")
.Input("num_output_spans: int32")
.Input("max_sentence_length: int32")
.Attr("sort_spans: bool")
.Output("output_span_indices: int32");

class ExtractSpansOp : public OpKernel {
public:
  explicit ExtractSpansOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("sort_spans", &_sort_spans));
  }

  void Compute(OpKernelContext* context) override {
    TTypes<float>::ConstMatrix span_scores = context->input(0).matrix<float>();
    TTypes<int32>::ConstMatrix candidate_starts = context->input(1).matrix<int32>();
    TTypes<int32>::ConstMatrix candidate_ends = context->input(2).matrix<int32>();
    TTypes<int32>::ConstVec num_output_spans = context->input(3).vec<int32>();
    int max_sentence_length = context->input(4).scalar<int32>()();

    int num_sentences = span_scores.dimension(0);
    int num_input_spans = span_scores.dimension(1);
    int max_num_output_spans = 0;
    for (int i = 0; i < num_sentences; i++) {
      if (num_output_spans(i) > max_num_output_spans) {
        max_num_output_spans = num_output_spans(i);
      }
    }

    Tensor* output_span_indices_tensor = nullptr;
    TensorShape output_span_indices_shape({num_sentences, max_num_output_spans});
    OP_REQUIRES_OK(context, context->allocate_output(0, output_span_indices_shape,
                                                     &output_span_indices_tensor));
    TTypes<int32>::Matrix output_span_indices = output_span_indices_tensor->matrix<int32>();

    std::vector<std::vector<int>> sorted_input_span_indices(num_sentences,
                                                            std::vector<int>(num_input_spans));
    for (int i = 0; i < num_sentences; i++) {
      std::iota(sorted_input_span_indices[i].begin(), sorted_input_span_indices[i].end(), 0);
      std::sort(sorted_input_span_indices[i].begin(), sorted_input_span_indices[i].end(),
                [&span_scores, &i](int j1, int j2) {
                  return span_scores(i, j2) < span_scores(i, j1);
                });
    }

    for (int l = 0; l < num_sentences; l++) {
      std::vector<int> top_span_indices;
      std::unordered_map<int, int> end_to_earliest_start;
      std::unordered_map<int, int> start_to_latest_end;

      int current_span_index = 0,
          num_selected_spans = 0;
      while (num_selected_spans < num_output_spans(l) && current_span_index < num_input_spans) {
        int i = sorted_input_span_indices[l][current_span_index];
        bool any_crossing = false;
        const int start = candidate_starts(l, i);
        const int end = candidate_ends(l, i);
        for (int j = start; j <= end; ++j) {
          auto latest_end_iter = start_to_latest_end.find(j);
          if (latest_end_iter != start_to_latest_end.end() && j > start && latest_end_iter->second > end) {
            // Given (), exists [], such that ( [ ) ]
            any_crossing = true;
            break;
          }
          auto earliest_start_iter = end_to_earliest_start.find(j);
          if (earliest_start_iter != end_to_earliest_start.end() && j < end && earliest_start_iter->second < start) {
            // Given (), exists [], such that [ ( ] )
            any_crossing = true;
            break;
          }
        }
        if (!any_crossing) {
          if (_sort_spans) {
            top_span_indices.push_back(i);
          } else {
            output_span_indices(l, num_selected_spans) = i;
          }
          ++num_selected_spans;
          // Update data struct.
          auto latest_end_iter = start_to_latest_end.find(start);
          if (latest_end_iter == start_to_latest_end.end() || end > latest_end_iter->second) {
            start_to_latest_end[start] = end;
          }
          auto earliest_start_iter = end_to_earliest_start.find(end);
          if (earliest_start_iter == end_to_earliest_start.end() || start < earliest_start_iter->second) {
            end_to_earliest_start[end] = start;
          }
        }
        ++current_span_index;
      }
      // Sort and populate selected span indices.
      if (_sort_spans) {
        std::sort(top_span_indices.begin(), top_span_indices.end(),
                  [&candidate_starts, &candidate_ends, &l] (int i1, int i2) {
                    if (candidate_starts(l, i1) < candidate_starts(l, i2)) {
                      return true;
                    } else if (candidate_starts(l, i1) > candidate_starts(l, i2)) {
                      return false;
                    } else if (candidate_ends(l, i1) < candidate_ends(l, i2)) {
                      return true;
                    } else if (candidate_ends(l, i1) > candidate_ends(l, i2)) {
                      return false;
                    } else {
                      return i1 < i2;
                    }
                  });
        for (int i = 0; i < num_output_spans(l); ++i) {
          output_span_indices(l, i) = top_span_indices[i];
        }
      }
      // Pad with the first span index.
      for (int i = num_selected_spans; i < max_num_output_spans; ++i) {
        output_span_indices(l, i) = output_span_indices(l, 0);
      }
    }
  }
private:
  bool _sort_spans;
};

REGISTER_KERNEL_BUILDER(Name("ExtractSpans").Device(DEVICE_CPU), ExtractSpansOp);
