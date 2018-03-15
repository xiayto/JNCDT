#pragma once
#include "Import.hpp"
#include "ModelConfig.hpp"
#include "DataModel.hpp"
#include <boost/progress.hpp>
#include<pthread.h>
#include<functional>
#include<thread>
#include <atomic>
#include <mutex>

using namespace std;
using namespace arma;

class Model
{
public:
	const DataModel&	data_model;
	const TaskType		task_type;
	const bool			be_deleted_data_model;

public:
	ModelLogging&		logging;

public:
	int	epos;
	int threadNum=24;
	int batchSize;
	thread **th=new thread*[threadNum];
	mutex fileLock;


public:
	Model(const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path)
		:data_model(*(new DataModel(dataset))), task_type(task_type),
		logging(*(new ModelLogging(logging_base_path))),
		be_deleted_data_model(true)
	{
		epos = 0;
		best_triplet_result = 0;
		std::cout << "Ready" << endl;

		logging.record() << "\t[Dataset]\t" << dataset.name;
		logging.record() << TaskTypeName(task_type);
	}

	Model(const Dataset& dataset,
		const string& file_zero_shot,
		const TaskType& task_type,
		const string& logging_base_path)
		:data_model(*(new DataModel(dataset))), task_type(task_type),
		logging(*(new ModelLogging(logging_base_path))),
		be_deleted_data_model(true)
	{
		epos = 0;
		best_triplet_result = 0;
		std::cout << "Ready" << endl;

		logging.record() << "\t[Dataset]\t" << dataset.name;
		logging.record() << TaskTypeName(task_type);
	}

	Model(const DataModel* data_model,
		const TaskType& task_type,
		ModelLogging* logging)
		:data_model(*data_model), logging(*logging), task_type(task_type),
		be_deleted_data_model(false)
	{
		epos = 0;
		best_triplet_result = 0;
	}

public:
	virtual double prob_triplets(const pair<pair<int, int>, int>& triplet) = 0;
	virtual void train_triplet(const pair<pair<int, int>, int>& triplet) = 0;
	virtual void train_batch(){};
	virtual void out_transe(int dimension){};
public:
	virtual void train(bool last_time = false)
	{
		++epos;

#pragma omp parallel for
		int nbatchs=50;
		batchSize=data_model.data_train.size()/nbatchs/threadNum;
		pair<pair<int, int>,int> triplet;
		train_batch();
		for(int i=0;i<nbatchs;i++){
			//thread thr[threadNum];
			for(int k=0;k<threadNum;k++)
				th[k]=new thread(
					[this]()->void{
							this->train_batch();
					}

				);
			for(int m=0;m<threadNum;m++)
				th[m]->join();

			for(int m=0;m<threadNum;m++)
				delete th[m];


		}
		/*for (auto i = data_model.data_train.begin(); i != data_model.data_train.end(); ++i)
		{

			train_triplet(*i);

		}*/

	}
	void sample(pair<pair<int, int>,int>& triplet){
		int index=rand_max(data_model.data_train.size());
		triplet=data_model.data_train[index];
	}

	void run(int total_epos)
	{
		logging.record() << "\t[Epos]\t" << total_epos;

		--total_epos;
		boost::progress_display	cons_bar(total_epos);
		while (total_epos-- > 0)
		{
			++cons_bar;
			train();

			if (task_type == TripletClassification)
				test_triplet_classification();
		}

		out_transe(100);
		train(true);
	}


public:
	double		best_triplet_result;
	double		best_link_mean;
	double		best_link_hitatten;
	double		best_link_fmean;
	double		best_link_fhitatten;
	int rand_max(int x)
	{
		int res = (rand()*rand())%x;
		while (res<0)
			res+=x;
		return res;
	}
	void reset()
	{
		best_triplet_result = 0;
		best_link_mean = 1e10;
		best_link_hitatten = 0;
		best_link_fmean = 1e10;
		best_link_fhitatten = 0;
	}

	void test(int hit_rank = 10)
	{
		logging.record();

		best_link_mean = 1e10;
		best_link_hitatten = 0;
		best_link_fmean = 1e10;
		best_link_fhitatten = 0;

		if (task_type == LinkPredictionHead || task_type == LinkPredictionTail || task_type == LinkPredictionRelation){
			if(task_type != LinkPredictionRelation){
				test_link_prediction_head(hit_rank);
				test_link_prediction_tail(hit_rank);
				/*test_link_prediction(hit_rank,true,0);
				test_link_prediction(hit_rank,false,0);*/
			}else{

				test_link_prediction_head(hit_rank);

			}



		}

		if (task_type == LinkPredictionHeadZeroShot || task_type == LinkPredictionTailZeroShot || task_type == LinkPredictionRelationZeroShot)
			test_link_prediction_zeroshot(hit_rank);
		else
			test_triplet_classification();
	}

public:
	void test_triplet_classification()
	{
		double real_hit = 0;
		for (auto r = 0; r < data_model.set_relation.size(); ++r)
		{
			vector<pair<double, bool>>	threshold_dev;
			for (auto i = data_model.data_dev_true.begin(); i != data_model.data_dev_true.end(); ++i)
			{
				if (i->second != r)
					continue;

				threshold_dev.push_back(make_pair(prob_triplets(*i), true));
			}
			for (auto i = data_model.data_dev_false.begin(); i != data_model.data_dev_false.end(); ++i)
			{
				if (i->second != r)
					continue;

				threshold_dev.push_back(make_pair(prob_triplets(*i), false));
			}

			sort(threshold_dev.begin(), threshold_dev.end());

			double threshold;
			double vari_mark = 0;
			int total = 0;
			int hit = 0;
			for (auto i = threshold_dev.begin(); i != threshold_dev.end(); ++i)
			{
				if (i->second == false)
					++hit;
				++total;

				if (vari_mark <= 2 * hit - total + data_model.data_dev_true.size())
				{
					vari_mark = 2 * hit - total + data_model.data_dev_true.size();
					threshold = i->first;
				}
			}

			double lreal_hit = 0;
			double lreal_total = 0;
			for (auto i = data_model.data_test_true.begin(); i != data_model.data_test_true.end(); ++i)
			{
				if (i->second != r)
					continue;

				++lreal_total;
				if (prob_triplets(*i) > threshold)
					++real_hit, ++lreal_hit;
			}

			for (auto i = data_model.data_test_false.begin(); i != data_model.data_test_false.end(); ++i)
			{
				if (i->second != r)
					continue;

				++lreal_total;
				if (prob_triplets(*i) <= threshold)
					++real_hit, ++lreal_hit;
			}

			//logging.record()<<data_model.relation_id_to_name.at(r)<<"\t"
			//	<<lreal_hit/lreal_total;
		}

		std::cout << epos << "\t Accuracy = "
			<< real_hit / (data_model.data_test_true.size() + data_model.data_test_false.size());
		best_triplet_result = max(
			best_triplet_result,
			real_hit / (data_model.data_test_true.size() + data_model.data_test_false.size()));
		std::cout << ", Best = " << best_triplet_result << endl;

		logging.record() << epos << "\t Accuracy = "
			<< real_hit / (data_model.data_test_true.size() + data_model.data_test_false.size())
			<< ", Best = " << best_triplet_result;

		std::cout.flush();
	}

	void test_link_prediction(int hit_rank = 10, bool is_head=true,const int part = 0)
	{
		double mean = 0;
		double hits = 0;
		double fmean = 0;
		double fhits = 0;
		double rmrr = 0;
		double fmrr = 0;
		double total = data_model.data_test_true.size();

		double arr_mean[20] = { 0 };
		double arr_total[5] = { 0 };

		ofstream fout;
		string path1;
		if(is_head)
			path1="./result/prediction_head.txt";
		else
			path1="./result/prediction_tail.txt";
		fout.open(path1.c_str(),ios::out|ios_base::trunc);

		for (auto i = data_model.data_test_true.begin(); i != data_model.data_test_true.end(); ++i)
		{
			++arr_total[data_model.relation_type[i->second]];
		}

		int cnt = 0;

		boost::progress_display cons_bar(data_model.data_test_true.size() / 100);

#pragma omp parallel for
		for (auto i = data_model.data_test_true.begin(); i != data_model.data_test_true.end(); ++i)
		{
			++cnt;
			if (cnt % 100 == 0)
			{
				++cons_bar;
			}

			pair<pair<int, int>, int> t = *i;
			int frmean = 0;
			int rmean = 0;
			double score_i = prob_triplets(*i);

			if (task_type == LinkPredictionRelation || part == 2)
			{
				for (auto j = 0; j != data_model.set_relation.size(); ++j)
				{
					t.second = j;

					if (score_i >= prob_triplets(t))
						continue;

					++rmean;

					if (data_model.check_data_all.find(t) == data_model.check_data_all.end())
						++frmean;
				}
			}
			else
			{
				for (auto j = 0; j != data_model.set_entity.size(); ++j)
				{
					//if (task_type == LinkPredictionHead || part == 1)
					if(is_head)
						t.first.first = j;
					else
						t.first.second = j;

					if (score_i >= prob_triplets(t))
						continue;

					++rmean;

					if (data_model.check_data_all.find(t) == data_model.check_data_all.end())
					{
						++frmean;
						//if (frmean > hit_rank)
						//	break;
					}
				}
			}

#pragma omp critical
			{
				if (frmean < hit_rank)
					++arr_mean[data_model.relation_type[i->second]];

				mean += rmean;
				fmean += frmean;
				rmrr += 1.0 / (rmean + 1);
				fmrr += 1.0 / (frmean + 1);

				if (rmean < hit_rank)
					++hits;
				if (frmean < hit_rank)
					++fhits;
			}
			int h_id=(*i).first.first;
			int r_id=(*i).second;
			int t_id=(*i).first.second;
			fout<<"head \t"<<data_model.entity_id_to_name[h_id]<<"\t relation \t"<<
					data_model.relation_id_to_name[r_id]<<"\t tail \t"<<
					data_model.entity_id_to_name[t_id]<<endl;
			fout<<"rmean \t"<<rmean<<"\t fmean \t"<<fmean<<"\t rhits \t"<<hits<<"\t fhits \t"<<fhits;
			fout<<endl;
		}
		fout.close();
		std::cout << endl;
		for (auto i = 1; i <= 4; ++i)
		{
			std::cout << i << ':' << arr_mean[i] / arr_total[i] << endl;
			logging.record() << i << ':' << arr_mean[i] / arr_total[i];
		}
		logging.record();

		best_link_mean = min(best_link_mean, mean / total);
		best_link_hitatten = max(best_link_hitatten, hits / total);
		best_link_fmean = min(best_link_fmean, fmean / total);
		best_link_fhitatten = max(best_link_fhitatten, fhits / total);

		std::cout << "Raw.BestMEANS = " << best_link_mean << endl;
		std::cout << "Raw.BestMRR = " << rmrr / total << endl;
		std::cout << "Raw.BestHITS = " << best_link_hitatten << endl;
		logging.record() << "Raw.BestMEANS = " << best_link_mean;
		logging.record() << "Raw.BestMRR = " << rmrr / total;
		logging.record() << "Raw.BestHITS = " << best_link_hitatten;

		std::cout << "Filter.BestMEANS = " << best_link_fmean << endl;
		std::cout << "Filter.BestMRR= " << fmrr / total << endl;
		std::cout << "Filter.BestHITS = " << best_link_fhitatten << endl;
		logging.record() << "Filter.BestMEANS = " << best_link_fmean;
		logging.record() << "Filter.BestMRR= " << fmrr / total;
		logging.record() << "Filter.BestHITS = " << best_link_fhitatten;

		std::cout.flush();
	}
	void test_link_prediction_head(int hit_rank = 10, const int part = 0){

		ofstream fout;
		string path1="./result/prediction_head.txt";
		fout.open(path1.c_str(),ios::out|ios_base::trunc);

		atomic<double> mean;
		atomic<double> hits;
		atomic<double> fmean;
		atomic<double> fhits;
		atomic<double> rmrr;
		atomic<double> fmrr;
		atomic<int> cnt;
		mean.store(0);hits.store(0);fmean.store(0);
		fhits.store(0);rmrr.store(0);fmrr.store(0);
		cnt.store(0);
		std::function<void(int,int)> func=[&](int s,int t)->void{
					for(int k=s;k<t;k++){
						if(k%100==0)
							cout<<"the "<<k<<"-th"<<endl;
						pair<pair<int, int>, int> t = data_model.data_test_true[k];
						atomic<double> frmean;
						atomic<double> rmean;
						int h_id=t.first.first;
						int r_id=t.second;
						int t_id=t.first.second;
						frmean.store(0);
						rmean.store(0);
						double score_i = prob_triplets(t);

						if (task_type == LinkPredictionRelation || part == 2)
						{
							for (auto j = 0; j != data_model.set_relation.size(); ++j)
							{
								t.second = j;

								if (score_i >= prob_triplets(t))
									continue;

								rmean=rmean+1;

								if (data_model.check_data_all.find(t) == data_model.check_data_all.end())
									frmean=frmean+1;
							}
						}
						else
						{
							for (auto j = 0; j != data_model.set_entity.size(); ++j)
							{
								//if (task_type != LinkPredictionHead || part == 1)
									t.first.first = j;
								/*else
									t.first.second = j;*/

								if (score_i >= prob_triplets(t))
									continue;

								rmean=rmean+1;

								if (data_model.check_data_all.find(t) == data_model.check_data_all.end())
								{
									frmean=frmean+1;
									//if (frmean > hit_rank)
									//	break;
								}
							}
						}

			#pragma omp critical
						{
							/*if (frmean < hit_rank)
								++arr_mean[data_model.relation_type[i->second]];
							 	 */
							mean =mean+ rmean;
							fmean =fmean+ frmean;
							rmrr =rmrr+ 1.0 / (rmean + 1);
							fmrr =fmrr+ 1.0 / (frmean + 1);

							if (rmean < hit_rank)
								hits=hits+1;
							if (frmean < hit_rank)
								fhits=fhits+1;

					}

					fout<<"head"<<"\t"<<data_model.entity_id_to_name[h_id]<<"\t"<<"relation"<<"\t"<<
												data_model.relation_id_to_name[r_id]<<"\t"<<"tail"<<"\t"<<
												data_model.entity_id_to_name[t_id]<<endl;
					fout<<"rmean"<<"\t"<<rmean<<"\t"<<"fmean"<<"\t"<<fmean<<"\t"<<"rhits"<<"\t"<<hits<<"\t fhits"<<"\t"<<
							fhits;
					fout<<endl;


			}
		};
		int testThreadNum=16;
		std::thread *threads[testThreadNum+1];
		double total = data_model.data_test_true.size();
		int psize=total/testThreadNum;
		int begin=0,end=psize;
		for(int i=0;i<testThreadNum;i++){

			threads[i]=new thread(func,begin,end);
			begin=end;
			end+=psize;
		}
		threads[testThreadNum]=new thread(func,begin,total);

		for(int i=0;i<testThreadNum+1;i++)
			threads[i]->join();

		for(int i=0;i<testThreadNum+1;i++)
			delete threads[i];

		logging.record();
		best_link_mean = min(best_link_mean, mean / total);
		best_link_hitatten = max(best_link_hitatten, hits / total);
		best_link_fmean = min(best_link_fmean, fmean / total);
		best_link_fhitatten = max(best_link_fhitatten, fhits / total);

		std::cout << "Raw.BestMEANS = " << best_link_mean << endl;
		std::cout << "Raw.BestMRR = " << rmrr / total << endl;
		std::cout << "Raw.BestHITS = " << best_link_hitatten << endl;
		logging.record() << "Raw.BestMEANS = " << best_link_mean;
		logging.record() << "Raw.BestMRR = " << rmrr / total;
		logging.record() << "Raw.BestHITS = " << best_link_hitatten;

		std::cout << "Filter.BestMEANS = " << best_link_fmean << endl;
		std::cout << "Filter.BestMRR= " << fmrr / total << endl;
		std::cout << "Filter.BestHITS = " << best_link_fhitatten << endl;
		logging.record() << "Filter.BestMEANS = " << best_link_fmean;
		logging.record() << "Filter.BestMRR= " << fmrr / total;
		logging.record() << "Filter.BestHITS = " << best_link_fhitatten;

		std::cout.flush();
		fout.close();

	}
	void test_link_prediction_tail(int hit_rank = 10, const int part = 0){
		reset();
		ofstream fout;
		string path1="./result/prediction_tail.txt";
		fout.open(path1.c_str(),ios::out|ios_base::trunc);
		atomic<double> mean;
		atomic<double> hits;
		atomic<double> fmean;
		atomic<double> fhits;
		atomic<double> rmrr;
		atomic<double> fmrr;
		atomic<int> cnt;
		mean.store(0);hits.store(0);fmean.store(0);
		fhits.store(0);rmrr.store(0);fmrr.store(0);
		cnt.store(0);
		std::function<void(int,int)> func=[&](int s,int t)->void{
					for(int k=s;k<t;k++){
						if(k%100==0)
							cout<<"the "<<k<<"-th"<<endl;
						pair<pair<int, int>, int> t = data_model.data_test_true[k];
						atomic<double> frmean;
						atomic<double> rmean;
						int h_id=t.first.first;
						int r_id=t.second;
						int t_id=t.first.second;
						frmean.store(0);
						rmean.store(0);
						double score_i = prob_triplets(t);

						if (task_type == LinkPredictionRelation || part == 2)
						{
							for (auto j = 0; j != data_model.set_relation.size(); ++j)
							{
								t.second = j;

								if (score_i >= prob_triplets(t))
									continue;

								rmean=rmean+1;

								if (data_model.check_data_all.find(t) == data_model.check_data_all.end())
									frmean=frmean+1;
							}
						}
						else
						{
							for (auto j = 0; j != data_model.set_entity.size(); ++j)
							{
								/*if (task_type == LinkPredictionHead || part == 1)
									t.first.first = j;
								else*/
									t.first.second = j;

								if (score_i >= prob_triplets(t))
									continue;

								rmean=rmean+1;

								if (data_model.check_data_all.find(t) == data_model.check_data_all.end())
								{
									frmean=frmean+1;
									//if (frmean > hit_rank)
									//	break;
								}
							}
						}

			#pragma omp critical
						{
							/*if (frmean < hit_rank)
								++arr_mean[data_model.relation_type[i->second]];
								 */
							mean =mean+ rmean;
							fmean =fmean+ frmean;
							rmrr =rmrr+ 1.0 / (rmean + 1);
							fmrr =fmrr+ 1.0 / (frmean + 1);

							if (rmean < hit_rank)
								hits=hits+1;
							if (frmean < hit_rank)
								fhits=fhits+1;

					}

					fout<<"head"<<"\t"<<data_model.entity_id_to_name[h_id]<<"\t"<<"relation"<<"\t"<<
							data_model.relation_id_to_name[r_id]<<"\t"<<"tail"<<"\t"<<
							data_model.entity_id_to_name[t_id]<<endl;
					fout<<"rmean"<<"\t"<<rmean<<"\t"<<"fmean"<<"\t"<<fmean<<"\t"<<"rhits"<<"\t"<<hits<<"\t fhits"<<"\t"<<
							fhits;
					fout<<endl;

			}
		};
		int testThreadNum=16;
		std::thread *threads[testThreadNum+1];
		double total = data_model.data_test_true.size();
		int psize=total/testThreadNum;
		int begin=0,end=psize;
		for(int i=0;i<testThreadNum;i++){

			threads[i]=new thread(func,begin,end);
			begin=end;
			end+=psize;
		}
		threads[testThreadNum]=new thread(func,begin,total);

		for(int i=0;i<testThreadNum+1;i++)
			threads[i]->join();

		for(int i=0;i<testThreadNum+1;i++)
			delete threads[i];

		logging.record();
		best_link_mean = min(best_link_mean, mean / total);
		best_link_hitatten = max(best_link_hitatten, hits / total);
		best_link_fmean = min(best_link_fmean, fmean / total);
		best_link_fhitatten = max(best_link_fhitatten, fhits / total);

		std::cout << "Raw.BestMEANS = " << best_link_mean << endl;
		std::cout << "Raw.BestMRR = " << rmrr / total << endl;
		std::cout << "Raw.BestHITS = " << best_link_hitatten << endl;
		logging.record() << "Raw.BestMEANS = " << best_link_mean;
		logging.record() << "Raw.BestMRR = " << rmrr / total;
		logging.record() << "Raw.BestHITS = " << best_link_hitatten;

		std::cout << "Filter.BestMEANS = " << best_link_fmean << endl;
		std::cout << "Filter.BestMRR= " << fmrr / total << endl;
		std::cout << "Filter.BestHITS = " << best_link_fhitatten << endl;
		logging.record() << "Filter.BestMEANS = " << best_link_fmean;
		logging.record() << "Filter.BestMRR= " << fmrr / total;
		logging.record() << "Filter.BestHITS = " << best_link_fhitatten;

		std::cout.flush();
		fout.close();

	}
public:
	void test_link_prediction_zeroshot(int hit_rank = 10, const int part = 0)
	{
		reset();

		double mean = 0;
		double hits = 0;
		double fmean = 0;
		double fhits = 0;
		double total = data_model.data_test_true.size();

		double arr_mean[20] = { 0 };
		double arr_total[5] = { 0 };

		cout << endl;

		for (auto i = data_model.data_test_true.begin(); i != data_model.data_test_true.end(); ++i)
		{
			if (i->first.first >= data_model.zeroshot_pointer
				&& i->first.second >= data_model.zeroshot_pointer)
			{
				++arr_total[3];
			}
			else if (i->first.first < data_model.zeroshot_pointer
				&& i->first.second >= data_model.zeroshot_pointer)
			{
				++arr_total[2];
			}
			else if (i->first.first >= data_model.zeroshot_pointer
				&& i->first.second < data_model.zeroshot_pointer)
			{
				++arr_total[1];
			}
			else
			{
				++arr_total[0];
			}
		}

		cout << "0 holds " << arr_total[0] << endl;
		cout << "1 holds " << arr_total[1] << endl;
		cout << "2 holds " << arr_total[2] << endl;
		cout << "3 holds " << arr_total[3] << endl;

		int cnt = 0;

#pragma omp parallel for
		for (auto i = data_model.data_test_true.begin(); i != data_model.data_test_true.end(); ++i)
		{
			++cnt;
			if (cnt % 100 == 0)
			{
				std::cout << cnt << ',';
				std::cout.flush();
			}

			pair<pair<int, int>, int> t = *i;
			int frmean = 0;
			int rmean = 0;
			double score_i = prob_triplets(*i);

			if (task_type == LinkPredictionRelationZeroShot || part == 2)
			{
				for (auto j = 0; j != data_model.set_relation.size(); ++j)
				{
					t.second = j;

					if (score_i >= prob_triplets(t))
						continue;

					++rmean;

					if (data_model.check_data_all.find(t) == data_model.check_data_all.end())
						++frmean;
				}
			}
			else
			{
				for (auto j = 0; j != data_model.set_entity.size(); ++j)
				{
					if (task_type == LinkPredictionHeadZeroShot || part == 1)
						t.first.first = j;
					else
						t.first.second = j;

					if (score_i >= prob_triplets(t))
						continue;

					++rmean;

					if (data_model.check_data_all.find(t) == data_model.check_data_all.end())
					{
						++frmean;
					}
				}
			}

#pragma omp critical
			{
				if (frmean < hit_rank)
				{
					if (i->first.first >= data_model.zeroshot_pointer
						&& i->first.second >= data_model.zeroshot_pointer)
					{
						++arr_mean[3];
					}
					else if (i->first.first < data_model.zeroshot_pointer
						&& i->first.second >= data_model.zeroshot_pointer)
					{
						++arr_mean[2];
					}
					else if (i->first.first >= data_model.zeroshot_pointer
						&& i->first.second < data_model.zeroshot_pointer)
					{
						++arr_mean[1];
					}
					else
					{
						++arr_mean[0];
					}
				}

				mean += rmean;
				fmean += frmean;
				if (rmean < hit_rank)
					++hits;
				if (frmean < hit_rank)
					++fhits;
			}
		}

		std::cout << endl;
		for (auto i = 0; i < 4; ++i)
		{
			std::cout << i << ':' << arr_mean[i] / arr_total[i] << endl;
			logging.record() << i << ':' << arr_mean[i] / arr_total[i];
		}
		logging.record();

		best_link_mean = min(best_link_mean, mean / total);
		best_link_hitatten = max(best_link_hitatten, hits / total);
		best_link_fmean = min(best_link_fmean, fmean / total);
		best_link_fhitatten = max(best_link_fhitatten, fhits / total);

		std::cout << "Raw.BestMEANS = " << best_link_mean << endl;
		std::cout << "Raw.BestHITS = " << best_link_hitatten << endl;
		logging.record() << "Raw.BestMEANS = " << best_link_mean;
		logging.record() << "Raw.BestHITS = " << best_link_hitatten;
		std::cout << "Filter.BestMEANS = " << best_link_fmean << endl;
		std::cout << "Filter.BestHITS = " << best_link_fhitatten << endl;
		logging.record() << "Filter.BestMEANS = " << best_link_fmean;
		logging.record() << "Filter.BestHITS = " << best_link_fhitatten;
	}

	virtual void draw(const string& filename, const int radius, const int id_relation) const
	{
		return;
	}

	virtual void draw(const string& filename, const int radius,
		const int id_head, const int id_relation)
	{
		return;
	}

	virtual void report(const string& filename) const
	{
		return;
	}
public:
	~Model()
	{
		logging.record();
		if (be_deleted_data_model)
		{
			delete &data_model;
			delete &logging;
		}
	}

public:
	int count_entity() const
	{
		return data_model.set_entity.size();
	}

	int count_relation() const
	{
		return data_model.set_relation.size();
	}

	const DataModel& get_data_model() const
	{
		return data_model;
	}

public:
	virtual void save(const string& filename)
	{
		cout << "BAD";
	}

	virtual void load(const string& filename)
	{
		cout << "BAD";
	}

	virtual vec entity_representation(int entity_id) const
	{
		cout << "BAD";
	}

	virtual vec relation_representation(int relation_id) const
	{
		cout << "BAD";
	}
};
