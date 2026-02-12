#include "annotate.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <random>
#include <assert.h>

const uint64_t max_next_seq = 0xffffffff;
using namespace std;

 
void annotate_cost(const string &trace_file, const int &n_extra_fields, bool is_offline, int offline_num,
    int64_t delta_ratio, int64_t fixed_byte, int64_t min_ratio, int64_t n_early_stop){
    auto expect_file = trace_file+".blc_d"+
        to_string(delta_ratio)+"_f"+to_string(fixed_byte)+"_m"+to_string(min_ratio); // wiki2018_100000000.tr_1.ant.blc_d20_f1024_m40
    ifstream cachefile(expect_file);
    if (cachefile.good()) {
        cerr<<"cost file has been annotated, so skip annotation"<<endl;
        return;
    }
    cerr<<"annotate_cost generate file:"<<trace_file<<endl;
    string generate_cost_file = trace_file.substr(0, trace_file.rfind("/")+1)+"generate.cost_d"+
        to_string(delta_ratio)+"_f"+to_string(fixed_byte)+"_m"+to_string(min_ratio);
    cerr<<"generated :annotate_cost generate file:"<<generate_cost_file<<endl;
    uint64_t t, id, size, write_cost, read_cost, next_seq;
    vector<uint64_t> extra_features(n_extra_fields, 0);
    uint64_t i=0;
    default_random_engine _generator = default_random_engine();
    uniform_int_distribution<std::size_t> _distribution = uniform_int_distribution<std::size_t>();
    int gdsf_cost, gdsf_time, gdsf_id, gdsf_size;

    unordered_map <uint64_t, uint64_t> _map_cost;

    vector<int64_t> offline_nums(offline_num,0);
    ifstream generateCostFile(generate_cost_file); //生成cost生成文件
    if (generateCostFile.good()) {
        cerr<<"generateCostFile:"+generate_cost_file+" has generated"<<endl;
    }else{
        ofstream outGenCostfile(generate_cost_file);
        if (!outGenCostfile) {
            cerr << "Exception opening/reading tmp file: " << generate_cost_file << endl;
            exit(-1);
        }
        ifstream infile(trace_file);
        if(!infile){
            cerr << "Exception opening/reading file: " << trace_file << endl;
            exit(-1);
        }

        
        // while(infile>> t >> id >> size) {
        while(true){
            for(int i=0;i<offline_num;i++){
                if(!(infile>>offline_nums[i])){
                    break; //break
                }
            }
            if(!(infile >> t >> id >> size)){
                break;
            }
            for (int j = 0; j < n_extra_fields; ++j)
                infile >> extra_features[j];

            int tmp_cost=-1;
            //计算cost 这里也要改下 object 的size
            // while (size > 1048576){
            //     size -= 1048576;
            // }

            auto iter = _map_cost.find(size);
            if(iter != _map_cost.end()){
                tmp_cost = iter->second;
            }else{
                int tmp_delta = _distribution(_generator) % delta_ratio;
                tmp_cost = (tmp_delta * fixed_byte + min_ratio * size)/1000;  //为每个请求生成cost
                _map_cost.insert({size, tmp_cost});
            }
            outGenCostfile << tmp_cost <<endl;
            if (!(i % 1000000))
                cerr<<"cost file writing: "<<i<<endl;

            if (n_early_stop!=-1 && i>=n_early_stop){
                break;
            }
            i++;
        }
        infile.close();
        outGenCostfile.close();
    }
    generateCostFile.close();



    auto timenow = chrono::system_clock::to_time_t(chrono::system_clock::now());
    auto tmp_file = trace_file + ".blc.tmp." + to_string(timenow);
    cerr<<"writing the annotated trace "<<tmp_file<<endl;
    uint64_t tmp_cost=0;

    ofstream outTracefile(tmp_file);
    if (!outTracefile) {
        cerr << "Exception opening/reading tmp file " << tmp_file << endl;
        exit(-1);
    }
    ifstream genCostFile(generate_cost_file);
    if(!genCostFile){
        cerr << "cost file open error: "<<generate_cost_file<<endl;
        exit(-1);
    }
    
    ifstream inTracefile(trace_file);
    if(!inTracefile){
        cerr << "Exception opening/reading file: " << trace_file << endl;
        exit(-1);
    }

    i=0;
    if(is_offline){
        while(true) {
            for(int i=0;i<offline_num;i++){
                if(!(inTracefile>>offline_nums[i])){
                    break; //break
                }
            }
            if(!(inTracefile >> t >> id >> size)){
                break;
            }
            for (int j = 0; j < n_extra_fields; ++j)
                inTracefile >> extra_features[j];

            if(!(genCostFile>>tmp_cost)){
                break;
            }

            outTracefile << tmp_cost;
            for(int i=0;i<offline_num;i++){
                outTracefile<<" "<<offline_nums[i];
            }
            ///size 要判断是否大于2GB  如果大于2GB  就要一直-2GB
            // while (size > 1048576){
            //     size -= 1048576;
            // }
            outTracefile <<" " << t << " " << id << " " << size;

            for (int j = 0; j < n_extra_fields; ++j)
                outTracefile << " " << extra_features[j];
            outTracefile <<endl;
            if (!(i % 1000000))
                cerr<<"writing: "<<i<<endl;
            i++;
        }
    }else{
        while(true) {
            ///size 要判断是否大于2GB  如果大于2GB  就要一直-2GB
            // while (size > 1048576){
            //     size -= 1048576;
            // }
            if(!(inTracefile>> t >> id >> size)){
                break;
            }
            for (int j = 0; j < n_extra_fields; ++j)
                inTracefile >> extra_features[j];
            if(!(genCostFile>>tmp_cost)){
                break;
            }

            // while (size > 1048576){
            //     size -= 1048576;
            // }

            outTracefile << tmp_cost << " " << t << " " << id << " " << size;
            for (int j = 0; j < n_extra_fields; ++j)
                outTracefile << " " << extra_features[j];
            outTracefile <<endl;
            if (!(i % 1000000))
                cerr<<"writing: "<<i<<endl;
            i++;
        }
    }

    genCostFile.close();
    outTracefile.close();
    inTracefile.close();

    if (rename(tmp_file.c_str(), expect_file.c_str())) {
        cerr << "Exception in renaming file from " << tmp_file << " to " << expect_file << " code: " << strerror(errno)
             << endl;
        return;
    }
}


void annotate(const string &trace_file, const int &n_extra_fields, int offline_num, int offline_threshold, int64_t n_early_stop) {
    /*
     * assume trace is less than max(uint32_t)
    * offline_num 记录每个请求的 未来的offline_num次请求对应index

     * */
    auto expect_file = trace_file+"_"+to_string(offline_num)+".ant";  // wiki_1.ant -->最终生成的trace文件
    ifstream cachefile(expect_file);
    if (cachefile.good()) {
        cerr<<"file has been annotated, so skip annotation"<<endl;
        return;
    }
    

    // in the first path, hold ids; in the reverse path, hold next_seq
    vector<vector<int64_t> > id_and_next_seq;
    id_and_next_seq.resize(offline_num);
    vector<uint64_t> id_seq;

    uint64_t id;
    int64_t i = 0;
    //not actually need t and size
    uint64_t t, size;
    vector<uint64_t> extra_features(n_extra_fields, 0);

    ifstream infile(trace_file);
    if (!infile) {
        cerr << "Exception opening/reading annotate original file " << trace_file << endl;
        exit(-1);
    }


    while(infile>> t >> id >> size) {  //输入的文件必须是t id size 的格式
        for (int j = 0; j < n_extra_fields; ++j)
            infile >> extra_features[j]; //占位符 没啥用
        if (!(++i%1000000))
            cerr<<"reading origin trace: "<<i<<endl;
        id_seq.emplace_back(id); //主要是为了记录读写的id
        if (n_early_stop != -1 && i >= n_early_stop){ //提前终止， 默认是-1
            break;
        }
    }

    uint64_t n_req = id_seq.size();
    std::cerr << "scanned trace n=" << n_req << std::endl;
    if (n_req > max_next_seq) { //能处理的读写序列最大范围
        cerr<<"Error: don't support more trace length than "<<max_next_seq<<endl;
        abort();
    }

    for(int i=0;i<offline_num;i++){ //看访问object的 后面offline_num个数据
        id_and_next_seq[i].resize(n_req);
    }
    unordered_map<uint64_t, uint32_t> last_seen; //某个id的下一次访问索引
    //先计算每个请求的下一次请求的索引
    for (i = n_req - 1; i >= 0; --i) { //所以是从后往前遍历
        uint64_t current_id = id_seq[i];
        auto lit = last_seen.find(current_id);
        if (lit != last_seen.end())
            id_and_next_seq[0][i] = lit->second; 
        else
            id_and_next_seq[0][i] = max_next_seq; //每个请求的下次请求index   没有下次请求就标记为一个最大值
        last_seen[current_id] = i;//request id 对应的下一次请求index
        if (!(i % 1000000))
            cerr<<"computing next t: "<<i<<endl;
    }
   for(int i=0;i<n_req;i++){
        for(int j=1;j<offline_num;j++){
            if(id_and_next_seq[0][i]>n_req){ //如果下一次请求为标记最大值，说明该object就请求了一次  所以后面的请求都标记为-1
                // id_and_next_seq[j][i]=-1;
                id_and_next_seq[j][i] = -1; //应该标记最大值吧? 不应该是-1
            }else{
                int nex_offline_num_seq = i;
                for(int m=0;m<=j;m++){
                    nex_offline_num_seq = id_and_next_seq[0][nex_offline_num_seq]; //递归的，当前请求的下一次请求 --> 下一次请求  --> 下一次请求 循环j次
                    if(nex_offline_num_seq > n_req){
                        break;
                    }
                }   
                id_and_next_seq[j][i] = nex_offline_num_seq;
                if(nex_offline_num_seq > offline_threshold +i){
                    id_and_next_seq[j][i]=-1;
                }
            }
        }
        if (!(i % 1000000))
            cerr<<"computing next offline_num t: "<<i<<endl;
   }

    string now;
    auto timenow = chrono::system_clock::to_time_t(chrono::system_clock::now());
    auto tmp_file = trace_file + ".ant.tmp." + to_string(timenow);
    cerr<<"writing the annotated trace "<<tmp_file<<endl;

    ofstream outfile(tmp_file);
    if (!outfile) {
        cerr << "Exception opening/reading tmp file " << tmp_file << endl;
        exit(-1);
    }

    infile.clear();
    infile.seekg(0, ios::beg);
    for (i = 0; i < (uint32_t) n_req; ++i) {
        infile >> t >> id >> size;
        for (int j = 0; j < n_extra_fields; ++j)
            infile >> extra_features[j];
        for(int j=0; j<offline_num;j++)
            outfile << id_and_next_seq[j][i]<<" ";
        ///size 要判断是否大于2GB  如果大于2GB  就要一直-2GB  这个处理是否合适?? 应该不用吧，原始代码好像没有处理
        // while (size > 1048576){
        //     size -= 1048576;
        // }
        outfile << t << " " << id << " " << size;
        for (int j = 0; j < n_extra_fields; ++j)
            outfile << " " << extra_features[j];
        outfile <<endl;
        if (!(i % 1000000))
            cerr<<"writing: "<<i<<endl;
    }

    outfile.close();

    if (rename(tmp_file.c_str(), expect_file.c_str())) {
        cerr << "Exception in renaming file from " << tmp_file << " to " << expect_file << " code: " << strerror(errno)
             << endl;
        return;
    }

}
