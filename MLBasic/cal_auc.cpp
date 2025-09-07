#include <vector>
#include <algorithm>
using namespace std;


struct Sample {    // 结构体
	int id;
	int label;
	double score;

	Sample(int id, int label, double score) : id(id), label(label), score(score) {}    // 构造函数
};


class Solution {
public:
	// 定义法计算AUC
	double auc_by_def(vector<Sample> samples) const {    // const表明函数是只读的
		vector<Sample> pos, neg;
		for (auto sample : samples) {
			if (sample.label == 1) pos.push_back(sample);
			else neg.push_back(sample);
		}
		int greaterCount = 0, equalCount = 0;
		int pos_cnt = pos.size(), neg_cnt = neg.size();
		for (int i = 0; i < pos_cnt; ++i) {
			for (int j = 0; j < neg_cnt; ++j) {
				if (pos[i].score > neg[j].score) greaterCount++;
				if (pos[i].score == neg[j].score) equalCount++;
			}
		}
		return (greaterCount * 1 + equalCount * 0.5) / (pos_cnt * neg_cnt);
	}


	// 公式法计算AUC
	double auc_by_formula(vector<Sample> samples) const {
		// 按照score从小到大排序
		std::sort(samples.begin(), samples.end(), [&](const Sample& s1, const Sample& s2) {
			return s1.score < s2.score;   // lambda表达式
		});

		int posCount = 0, negCount = 0;
		double posRankSum = 0;    // 所有正样本的排名和

		int i = 0;
		while (i < samples.size()) {
			// 查找相同得分的连续样本范围 (索引, 从0开始)
			int start = i;
			double cur_score = samples[i].score;
			while (i < samples.size() && samples[i].score == cur_score) {
				i++;
			}
			int end = i - 1;

			double avgRank = (start + end) / 2.0 + 1;    // 排名, 从1开始

			// 处理当前得分区间内的每一个样本
			for (int j = start; j <= end; j++) {
				if (samples[j].label == 1) {
					posCount++;
					posRankSum += avgRank;
				}
				else {
					negCount++;
				}
			}

		}

		return (posRankSum - posCount * (posCount + 1) / 2.0) / (posCount * negCount);
	}

};


int main() {
	vector<vector<double>> data1 = { {1, 1, 0.8}, {2, 1, 0.7}, {3, 0, 0.5}, {4, 0, 0.5}, {5, 1, 0.5}, {6, 1, 0.5}, {7, 0, 0.3} };
	vector<vector<double>> data2 = { {1, 0, 0.1}, {2, 0, 0.4}, {3, 1, 0.4}, {4, 1, 0.8} };

	Solution s;
	vector<vector<vector<double>>> dataset = { data1, data2 };
	for (const auto& data : dataset) {
		vector<Sample> samples;
		for (int i = 0; i < data.size(); ++i) {
			samples.emplace_back(data[i][0], data[i][1], data[i][2]);
		}
		double auc_def = s.auc_by_def(samples);
		double auc_formula = s.auc_by_formula(samples);
		printf("auc_def = %.3f, auc_formula = %.3f\n", auc_def, auc_formula);    // 保留3位小数
	}

	return 0;
}