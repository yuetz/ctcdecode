#include "ctc_beam_search_decoder.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <utility>

#include "decoder_utils.h"
#include "ThreadPool.h"
#include "fst/fstlib.h"
#include "path_trie.h"

using FSTMATCH = fst::SortedMatcher<fst::StdVectorFst>;

std::vector<std::pair<double, Output>> ctc_beam_search_decoder(
    const std::vector<std::vector<double>> &probs_seq,
    const std::vector<std::string> &vocabulary,
    size_t beam_size,
    double cutoff_prob,
    size_t cutoff_top_n,
    size_t blank_id,
    int log_input,
    Scorer *ext_scorer) {
  // dimension check
  size_t num_time_steps = probs_seq.size();
  for (size_t i = 0; i < num_time_steps; ++i) {
    VALID_CHECK_EQ(probs_seq[i].size(),
                   vocabulary.size(),
                   "The shape of probs_seq does not match with "
                   "the shape of the vocabulary");
  }

  // assign blank id
  // size_t blank_id = vocabulary.size();

  // assign space id
  auto it = std::find(vocabulary.begin(), vocabulary.end(), " ");
  int space_id = it - vocabulary.begin();
  // if no space in vocabulary
  if ((size_t)space_id >= vocabulary.size()) {
    space_id = -2;
  }

  // init prefixes' root
  PathTrie root;
  root.score = root.log_prob_b_prev = 0.0;
  std::vector<PathTrie *> prefixes;
  prefixes.push_back(&root);

  if (ext_scorer != nullptr && !ext_scorer->is_character_based()) {
    auto fst_dict = static_cast<fst::StdVectorFst *>(ext_scorer->dictionary);
    fst::StdVectorFst *dict_ptr = fst_dict->Copy(true);
    root.set_dictionary(dict_ptr);
    auto matcher = std::make_shared<FSTMATCH>(*dict_ptr, fst::MATCH_INPUT);
    root.set_matcher(matcher);
  }

  // prefix search over time
  for (size_t time_step = 0; time_step < num_time_steps; ++time_step) {
    auto &prob = probs_seq[time_step];

    float min_cutoff = -NUM_FLT_INF;
    bool full_beam = false;
    if (ext_scorer != nullptr) {
      size_t num_prefixes = std::min(prefixes.size(), beam_size);
      std::sort(
          prefixes.begin(), prefixes.begin() + num_prefixes, prefix_compare);
      float blank_prob = log_input ? prob[blank_id] : std::log(prob[blank_id]);
      // 为什么要减去beta？
      min_cutoff = prefixes[num_prefixes - 1]->score +
                   blank_prob - std::max(0.0, ext_scorer->beta);
      full_beam = (num_prefixes == beam_size);
    }

    std::vector<std::pair<size_t, float>> log_prob_idx =
        get_pruned_log_probs(prob, cutoff_prob, cutoff_top_n, log_input);
    // loop over chars
    for (size_t index = 0; index < log_prob_idx.size(); index++) {
      auto c = log_prob_idx[index].first;
      auto log_prob_c = log_prob_idx[index].second;

      for (size_t i = 0; i < prefixes.size() && i < beam_size; ++i) {
        auto prefix = prefixes[i];
        if (full_beam && log_prob_c + prefix->score < min_cutoff) {
          break;
        }
        // blank
        if (c == blank_id) {
          // c 为blank时，只更新blank的概率
          // 因为 blank + blank = blank
          // no blank + blank = blank
          // 所以 使用 p_c * (p_b + p_nb)
          prefix->log_prob_b_cur =
              log_sum_exp(prefix->log_prob_b_cur, log_prob_c + prefix->score);
          continue;
        }
        // repeated character
        if (c == prefix->character) {
          // c 等于prefix的最后一个字符时，只更新no blank的概率
          // p_c * p_nb
          prefix->log_prob_nb_cur = log_sum_exp(
              prefix->log_prob_nb_cur, log_prob_c + prefix->log_prob_nb_prev);
        }
        // get new prefix
        auto prefix_new = prefix->get_path_trie(c, time_step, log_prob_c);

        if (prefix_new != nullptr) {
          float log_p = -NUM_FLT_INF;

          if (c == prefix->character &&
              prefix->log_prob_b_prev > -NUM_FLT_INF) {
            // c 等于prefix的最后一个字符 且  p_nb_pre 不等于0，则新的p_c = p_c * p_nb_pre
            log_p = log_prob_c + prefix->log_prob_b_prev;
          } else if (c != prefix->character) {
            log_p = log_prob_c + prefix->score;
          }

          // language model scoring
          if (ext_scorer != nullptr &&
              (c == space_id || ext_scorer->is_character_based())) {

            // 当前字符是 空格 或者 使用基于char的LM时 考虑LM加权
            PathTrie *prefix_to_score = nullptr;
            // skip scoring the space
            if (ext_scorer->is_character_based()) {
              prefix_to_score = prefix_new;
            } else {
              prefix_to_score = prefix;
            }

            float score = 0.0;
            std::vector<std::string> ngram;
            // 根据当前节点，往父节点找 找到 ngram对应的单词或者 字母
            ngram = ext_scorer->make_ngram(prefix_to_score);
            score = ext_scorer->get_log_cond_prob(ngram) * ext_scorer->alpha;
            // 更新后的概率为p * p_text^alpha * beta
            log_p += score;
            log_p += ext_scorer->beta;
          }
          prefix_new->log_prob_nb_cur =
              log_sum_exp(prefix_new->log_prob_nb_cur, log_p);
        }
      }  // end of loop over prefix
    }    // end of loop over vocabulary


    prefixes.clear();
    // update log probs
    // 经root节点递归push到prefixes中，同时更新每个prefix节点的probs
    // 更新后的 p_b_cur 和 p_nb_cur 都为0
    root.iterate_to_vec(prefixes);

    // only preserve top beam_size prefixes
    if (prefixes.size() >= beam_size) {
      std::nth_element(prefixes.begin(),
                       prefixes.begin() + beam_size,
                       prefixes.end(),
                       prefix_compare);
      for (size_t i = beam_size; i < prefixes.size(); ++i) {
        prefixes[i]->remove();
      }
    }
  }  // end of loop over time

  // score the last word of each prefix that doesn't end with space
  if (ext_scorer != nullptr && !ext_scorer->is_character_based()) {
    for (size_t i = 0; i < beam_size && i < prefixes.size(); ++i) {
      auto prefix = prefixes[i];
      if (!prefix->is_empty() && prefix->character != space_id) {
        float score = 0.0;
        std::vector<std::string> ngram = ext_scorer->make_ngram(prefix);
        score = ext_scorer->get_log_cond_prob(ngram) * ext_scorer->alpha;
        score += ext_scorer->beta;
        prefix->score += score;
      }
    }
  }

  size_t num_prefixes = std::min(prefixes.size(), beam_size);
  std::sort(prefixes.begin(), prefixes.begin() + num_prefixes, prefix_compare);

  // compute aproximate ctc score as the return score, without affecting the
  // return order of decoding result. To delete when decoder gets stable.
  for (size_t i = 0; i < beam_size && i < prefixes.size(); ++i) {
    double approx_ctc = prefixes[i]->score;
    if (ext_scorer != nullptr) {
      std::vector<int> output;
      std::vector<int> timesteps;
      prefixes[i]->get_path_vec(output, timesteps);
      auto prefix_length = output.size();
      auto words = ext_scorer->split_labels(output);
      // remove word insert
      // beta 的数目为什么是prefix_length?而不是等于alpha对应的数目？
      approx_ctc = approx_ctc - prefix_length * ext_scorer->beta;
      // remove language model weight:
      approx_ctc -= (ext_scorer->get_sent_log_prob(words)) * ext_scorer->alpha;
    }
    prefixes[i]->approx_ctc = approx_ctc;
  }

  return get_beam_search_result(prefixes, beam_size);
}


std::vector<std::vector<std::pair<double, Output>>>
ctc_beam_search_decoder_batch(
    const std::vector<std::vector<std::vector<double>>> &probs_split,
    const std::vector<std::string> &vocabulary,
    size_t beam_size,
    size_t num_processes,
    double cutoff_prob,
    size_t cutoff_top_n,
    size_t blank_id,
    int log_input,
    Scorer *ext_scorer) {
  VALID_CHECK_GT(num_processes, 0, "num_processes must be nonnegative!");
  // thread pool
  ThreadPool pool(num_processes);
  // number of samples
  size_t batch_size = probs_split.size();

  // enqueue the tasks of decoding
  std::vector<std::future<std::vector<std::pair<double, Output>>>> res;
  for (size_t i = 0; i < batch_size; ++i) {
    res.emplace_back(pool.enqueue(ctc_beam_search_decoder,
                                  probs_split[i],
                                  vocabulary,
                                  beam_size,
                                  cutoff_prob,
                                  cutoff_top_n,
                                  blank_id,
                                  log_input,
                                  ext_scorer));
  }

  // get decoding results
  std::vector<std::vector<std::pair<double, Output>>> batch_results;
  for (size_t i = 0; i < batch_size; ++i) {
    batch_results.emplace_back(res[i].get());
  }
  return batch_results;
}
