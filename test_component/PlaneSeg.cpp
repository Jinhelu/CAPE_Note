/*
 * Copyright 2018 Pedro Proenza <p.proenca@surrey.ac.uk> (University of Surrey)
 *
 */

#include "PlaneSeg.h"

PlaneSeg::PlaneSeg(Eigen::MatrixXf & cloud_array, int cell_id, int nr_pts_per_cell, int cell_width)
{
	nr_pts = 0;
	min_nr_pts = nr_pts_per_cell/2;
	int offset = cell_id*nr_pts_per_cell;
	int cell_height = nr_pts_per_cell/cell_width;
	x_acc = 0; y_acc = 0; z_acc = 0;
	xx_acc = 0; yy_acc = 0; zz_acc = 0;
	xy_acc = 0; xz_acc = 0; yz_acc = 0;
	int nan_counter = 0;
	double max_diff = 100;
	int max_pt_id = offset+nr_pts_per_cell;
	planar = true;
	//block(i,j,p,q) 从矩阵中取一个起始于(i,j)p行q列的矩阵块
	Eigen::MatrixXf Z_matrix = cloud_array.block(offset,2,nr_pts_per_cell,1);

	//此参数不适用于3D雷达，统计z轴坐标大于0的点，即统计有效点
	nr_pts =  (Z_matrix.array()>0).count();//栅格内有效点个数

	if (nr_pts<min_nr_pts){
		planar = false;
		return;
	}

	Eigen::MatrixXf X_matrix = cloud_array.block(offset,0,nr_pts_per_cell,1);
	Eigen::MatrixXf Y_matrix = cloud_array.block(offset,1,nr_pts_per_cell,1);

	// Check for discontinuities using cross search，使用十字交叉搜索检查cell的不连续性
	//从一个cell内的中间位置水平扫描
	//在一个栅格内进行操作，将i定义为栅格总数的一半，j为i下一行同一列的索引
	int jumps_counter = 0;
	int i = cell_width*(cell_height/2);//水平方向中点的索引值
	int j = i+cell_width;
	float z(0), z_last(max(Z_matrix(i),Z_matrix(i+1))); /* handles missing pixels on the borders*/
	i++;
	while(i<j){
		z = Z_matrix(i);
		if(z>0 && abs(z-z_last)<max_diff){
			z_last = z;
		}else{
			if(z>0)
				jumps_counter++;
		}
		i++;
	}
	if (jumps_counter>1){
		planar = false;
		return;
	}

	// Scan vertically through the middle
	i = cell_width/2;
	j = nr_pts_per_cell-i;
	z_last = max(Z_matrix(i),Z_matrix(i+cell_width));  /* handles missing pixels on the borders*/
	i=i+cell_width;
	jumps_counter = 0;
	while(i<j){
		z = Z_matrix(i);
		if(z>0 && abs(z-z_last)<max_diff){
			z_last = z;
		}else{
			if(z>0)
				jumps_counter++;
		}
		i+=cell_width;
	}
	if (jumps_counter>1){
		planar = false;
		return;
	}

	x_acc = X_matrix.sum();
	y_acc = Y_matrix.sum();
	z_acc = Z_matrix.sum();
	xx_acc = (X_matrix.array()*X_matrix.array()).sum();
	yy_acc = (Y_matrix.array()*Y_matrix.array()).sum();
	zz_acc = (Z_matrix.array()*Z_matrix.array()).sum();
	xy_acc = (X_matrix.array()*Y_matrix.array()).sum();
	xz_acc = (X_matrix.array()*Z_matrix.array()).sum();
	yz_acc = (Y_matrix.array()*Z_matrix.array()).sum();

	if (planar){
		fitPlane();
		if (MSE>pow(DEPTH_SIGMA_COEFF*mean[2]*mean[2]+DEPTH_SIGMA_MARGIN,2)){
			planar = false;
		}
	}
}

void PlaneSeg::expandSegment(PlaneSeg * plane_seg){
	x_acc += plane_seg->x_acc; y_acc += plane_seg->y_acc; z_acc += plane_seg->z_acc;
	xx_acc += plane_seg->xx_acc; yy_acc += plane_seg->yy_acc; zz_acc += plane_seg->zz_acc;
	xy_acc += plane_seg->xy_acc; xz_acc += plane_seg->xz_acc; yz_acc += plane_seg->yz_acc;
	nr_pts += plane_seg->nr_pts;
}

void PlaneSeg::clearPoints()
{
	x_acc = 0; y_acc = 0; z_acc = 0;
	xx_acc = 0; yy_acc = 0; zz_acc = 0;
	xy_acc = 0; xz_acc = 0; yz_acc = 0;
	nr_pts = 0;
}

void PlaneSeg::fitPlane()
{
	mean[0] = x_acc/nr_pts;//x坐标值的平均
	mean[1] = y_acc/nr_pts;//y坐标值的平均
	mean[2] = z_acc/nr_pts;//z坐标值的平均
	//计算协方差矩阵 E(XY)-E(X)E(Y),下面的写法相当于协方差矩阵乘了一个nr_pts
	double cov[3][3] = {
		{xx_acc - x_acc*x_acc/nr_pts, xy_acc - x_acc*y_acc/nr_pts, xz_acc - x_acc*z_acc/nr_pts},
		{0, yy_acc - y_acc*y_acc/nr_pts, yz_acc - y_acc*z_acc/nr_pts},
		{0, 0, zz_acc - z_acc*z_acc/nr_pts }
	};
	cov[1][0]=cov[0][1]; cov[2][0]=cov[0][2]; cov[2][1]=cov[1][2];

	//使用QR分解求对称矩阵的特征值和特征向量
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(Eigen::Map<Eigen::Matrix3d>(cov[0], 3, 3) );
	//得到的一个从小到大排列的3x1特征值向量和与特征值对应的3x3特征向量
	Eigen::VectorXd v = es.eigenvectors().col(0);//代码此处取的是最小特征值对应的特征向量,特征向量已经被归一化
	//认为沿z轴方向的数据方差最小，并用mse和score来定量评价这个离散度的大小
	d = - (v[0]*mean[0]+v[1]*mean[1]+v[2]*mean[2]);//加负号，是为了让法向量方向都朝外(朝外发散)
	//求cell内数据所在曲面的法向量
	if(d>0) {
		normal[0]=v[0];
		normal[1]=v[1];
		normal[2]=v[2];
	} else {
		normal[0]=-v[0];
		normal[1]=-v[1];
		normal[2]=-v[2];
		d = -d;
	} 
	MSE = es.eigenvalues()[0]/nr_pts;//使用最小特征值的大小来表示均方误差
	score = es.eigenvalues()[1]/es.eigenvalues()[0];//表示次大主轴与最小主轴之间的比值
}

PlaneSeg::~PlaneSeg(void)
{
}
