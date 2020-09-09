/*
 * Copyright 2018 Pedro Proenza <p.proenca@surrey.ac.uk> (University of Surrey)
 *
 */

#include "Histogram.h"

Histogram::Histogram(int nr_bins_per_coord)
{
	this->nr_bins_per_coord = nr_bins_per_coord;
	this->nr_bins= nr_bins_per_coord*nr_bins_per_coord;
	this->H.assign(nr_bins,0);//赋值nr_bins个0到vector容器中
}

void Histogram::initHistogram(Eigen::MatrixXd & P, vector<bool> & Flags){

	int nr_points = P.rows();
	this->nr_points = nr_points;
	this->B.assign(nr_points,-1); 

	// 设置直方图两个变量的阈值
	// Polar angle [0 pi]
	double min_X(0), max_X(3.14);
	// Azimuth angle [-pi pi]
	double min_Y(-3.14), max_Y(3.14);

	// 填充直方图
	int X_q, Y_q;
	for (int i=0; i<nr_points; i++){
		if (Flags[i]){
			X_q = (nr_bins_per_coord-1)*(P(i,0)-min_X)/(max_X-min_X);
			// Dealing with degeneracy，极角在阈值范围之外，则将方位角置为0
			if (X_q>0){
				Y_q = (nr_bins_per_coord-1)*(P(i,1)-min_Y)/(max_Y-min_Y);
			}else{
				Y_q = 0;
			}
			int bin  = Y_q*nr_bins_per_coord + X_q;
			B[i] = bin;
			H[bin]++;
		}
	}
}

vector<int> Histogram::getPointsFromMostFrequentBin(){

	vector<int> point_ids;
	int most_frequent_bin = -1;
	int max_nr_occurrences = 0;
	for (int i=0; i<nr_bins; i++){
		if (H[i]>max_nr_occurrences){
			most_frequent_bin = i;
			max_nr_occurrences = H[i];
		}
	}//一次遍历找到最大的 H[bin]，将bin定义为最大出现次数的bin，对应H[bin]为出现次数 
	if(max_nr_occurrences>0){
		for (int i=0; i<nr_points; i++){
			if (B[i]==most_frequent_bin){
				point_ids.push_back(i);
			}
		}
	}//如果存在一个最大出现的次数，将所有该直方图通道内的cell下标号存入一个vector并返回
	return point_ids;
}

void Histogram::removePoint(int point_id){
	H[B[point_id]]--;
	B[point_id] = -1;
}

Histogram::~Histogram(void)
{
}
