#include "belady_sample_size_cost_p.h"
#include "utils.h"

using namespace std;

bool BeladySampleSizeCostCacheP::lookup(const SimpleRequest &_req) {
    auto & req = dynamic_cast<const AnnotatedRequest &>(_req);
    current_t = req.seq;
    auto it = key_map.find(req.id);
    if (it != key_map.end()) {
        uint32_t &pos_idx = it->second;
        meta_holder[pos_idx].update(req.seq, req.next_seq);
        return true;
    }
    return false;
}

int BeladySampleSizeCostCacheP::csb(double cs) {
    if (!std::isfinite(cs)) return 1000;
    if (cs < 0) return 0;
    if (cs >= 5.0) return 1000;

    int i = static_cast<int>(std::floor(cs * 200.0 + 1e-9));
    if (i < 0) i = 0;
    if (i > 999) i = 999;
    return i;
}

void BeladySampleSizeCostCacheP::admit(const SimpleRequest &_req) {
    auto & req = static_cast<const AnnotatedRequest &>(_req);
    const uint64_t & size = req.size;
    if (size > _cacheSize) {
        LOG("L", _cacheSize, req.id, size);
        return;
    }
    double cs_ratio = (double)req.cost / (double)req.size;
    size_t idx = csb(cs_ratio);

    if(is_bypss_cold_data && (req.next_seq - current_t > memory_window)){
        if (g_pass_bins.size() <= idx) {
                g_pass_bins.resize(1010, 0);
            }
            g_pass_bins[idx]++;

        return;
    }

    auto it = key_map.find(req.id);
    if (it == key_map.end()) {
       if( rank_admit(req) ) {

        if (g_admit_bins.size() <= idx) {
                g_admit_bins.resize(1010, 0);
            }
            g_admit_bins[idx]++;

            key_map.insert({req.id, (uint32_t) meta_holder.size()});
            meta_holder.emplace_back(req.id, req.size, req.seq, req.next_seq, req.cost);
            _currentSize += size;
        }
        else{
            if (g_pass_bins.size() <= idx) {
                g_pass_bins.resize(1010, 0);
            }
            g_pass_bins[idx]++;

            admit_bypass_num++;
        }
    }
    while (_currentSize > _cacheSize) {
        eviction_num++;
        evict();
    }
}

pair<uint64_t, uint32_t> BeladySampleSizeCostCacheP::rank() {
    vector<pair<uint64_t, uint32_t >> beyond_boundary_key_pos;
    long double max_future_interval = 0.0;
    uint64_t max_key;
    uint32_t max_pos;

    uint n_sample = min(sample_rate, (uint32_t) meta_holder.size());

    for (uint32_t i = 0; i < n_sample; i++) {
        uint32_t pos = (i + _distribution(_generator)) % meta_holder.size();
        auto &meta = meta_holder[pos];

        long double future_interval;
        if (meta._future_timestamp - current_t <= threshold) {
            future_interval = static_cast<double>((meta._future_timestamp - current_t)*meta._size) / meta._cost;
        } else {
            beyond_boundary_key_pos.emplace_back(pair(meta._key, pos));
            continue;
        }

        if (future_interval > max_future_interval) {
            max_future_interval = future_interval;
            max_key = meta._key;
            max_pos = pos;
        }
    }

    if(max_rank < max_future_interval){
        max_rank = max_future_interval;
    }
    if(total_max_rank < max_future_interval){
        total_max_rank = max_future_interval;
    }
    if(min_rank < 0 || min_rank > max_future_interval){
        min_rank = max_future_interval;
    }
    if(total_min_rank < 0 || total_min_rank > max_future_interval){
        total_min_rank = max_future_interval;
    }

    if (beyond_boundary_key_pos.empty()) {
        return {max_key, max_pos};
    } else {
        total_exceed_threshold++;
        exceed_threshold++;
        auto rand_id = _distribution(_generator) % beyond_boundary_key_pos.size();
        auto &item = beyond_boundary_key_pos[rand_id];
        return {item.first, item.second};
    }
}

void BeladySampleSizeCostCacheP::evict() {
    auto epair = rank();
    uint64_t & key = epair.first;
    uint32_t & old_pos = epair.second;
    _currentSize -= meta_holder[old_pos]._size;

    double evict_rate_  = meta_holder[old_pos]._cost /(double)meta_holder[old_pos]._size;
    if(evict_max_size_cost_rate < 0 || evict_max_size_cost_rate < evict_rate_){
        evict_max_size_cost_rate = evict_rate_;
    }
    if(evict_min_size_cost_rate < 0 || evict_min_size_cost_rate > evict_rate_){
        evict_min_size_cost_rate = evict_rate_;
    }

    if(is_write_file){
        evict_size_cost_ratios.insert(evict_rate_);
    }

    if(total_evict_min_size_cost_rate < 0 || total_evict_min_size_cost_rate > evict_rate_){
        total_evict_min_size_cost_rate = evict_rate_;
    }
    if(total_evict_max_size_cost_rate < 0 || total_evict_max_size_cost_rate < evict_rate_){
        total_evict_max_size_cost_rate = evict_rate_;
    }

    uint32_t activate_tail_idx = meta_holder.size() - 1;
    if (old_pos !=  activate_tail_idx) {
        meta_holder[old_pos] = meta_holder[activate_tail_idx];
        key_map.find(meta_holder[activate_tail_idx]._key)->second = old_pos;
    }
    meta_holder.pop_back();

    key_map.erase(key);
}

bool BeladySampleSizeCostCacheP::rank_admit(const SimpleRequest &_req) {
    auto &req = static_cast<const AnnotatedRequest &>(_req);

    long double new_request_score = static_cast<long double>((req.next_seq - current_t) * req.size) / req.cost;
    
    if (_currentSize + req.size <= _cacheSize) {
        return true;
    }

    if (req.next_seq - current_t > threshold) {
        return false;
    }

    if ((req.cost/ req.size) > cost_threshold){
        return true;
    }

    uint n_sample = min(sample_rate, (uint32_t)meta_holder.size());
    long double max_sample_score = std::numeric_limits<long double>::min();
    
    for (uint32_t i = 0; i < n_sample; i++) {
        uint32_t pos = (i + _distribution(_generator)) % meta_holder.size();
        auto &meta = meta_holder[pos];

        long double sample_score = static_cast<long double>((meta._future_timestamp - current_t) * meta._size) / meta._cost;

        if (sample_score > max_sample_score) {
            max_sample_score = sample_score;
        }

    }

    if (new_request_score < max_sample_score) {
        return true;
    }

    return false;
}
