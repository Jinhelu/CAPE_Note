#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/filter.h>
#include <pcl/console/parse.h>
#include <pcl/common/common_headers.h>

#include <string>
#include <iostream>

#include <opencv2/opencv.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include "CAPE.h"

using namespace std;
using namespace Eigen;

int nFRAMES = 1;//输入pcd文件的帧数--849
typedef pcl::PointXYZI PointType;

//参数采用了velodyne16线雷达的参数
const int _vertical_scans = 16;
const int _horizontal_scans = 1800;
int vertical_angle_top = 15;//角度
int vertical_angle_bottom = -15;
const int cloud_size = _vertical_scans * _horizontal_scans;

float DEG_TO_RAD = M_PI/180;
float _ang_bottom = -(vertical_angle_bottom - 0.1) * DEG_TO_RAD;//弧度,加负号是为了取绝对值
float _ang_resolution_X = (M_PI*2) / (_horizontal_scans);
float _ang_resolution_Y =  DEG_TO_RAD*(vertical_angle_top - vertical_angle_bottom) / float(_vertical_scans-1);

typedef struct seg_msg
{
  //缺少一个时间戳信息，待补充
  int32_t startRingIndex[_vertical_scans];
  int32_t endRingIndex[_vertical_scans];

  float startOrientation;
  float endOrientation;
  float orientationDiff;

  bool    segmentedCloudGroundFlag[cloud_size]; //true - ground point, false - other points
  uint32_t  segmentedCloudColInd[cloud_size]; // point column index in range image
  float segmentedCloudRange[cloud_size]; // point range 
}seg_msg;

seg_msg _seg_msg;

/*------将点云结构化所用到的变量--------*/
Eigen::MatrixXf _range_mat;   // 深度图像的深度矩阵
Eigen::MatrixXf cloud_array_organized(_vertical_scans * _horizontal_scans,3);

pcl::PointCloud<PointType>::Ptr _laser_cloud_in(new pcl::PointCloud<PointType>);
pcl::PointCloud<PointType>::Ptr _full_cloud(new pcl::PointCloud<PointType>);
//pcl::PointCloud<PointType>::Ptr _full_info_cloud(new pcl::PointCloud<PointType>);

/*------CAPE算法相关参数--------*/
bool done = false;
const int PATCH_SIZE = 4;//cell的尺寸，像素点为单位
int nr_horizontal_cells = _horizontal_scans/PATCH_SIZE;//cell的列数
int nr_vertical_cells = _vertical_scans/PATCH_SIZE;//cell的行数
float COS_ANGLE_MAX = cos(M_PI/12);
float MAX_MERGE_DIST = 50.0f;
bool cylinder_detection= false;//确定是否检测圆柱体
CAPE *plane_detector;

/*------将每一帧pcd文件的绝对地址存入一个vector--------*/
vector<string> loadData()
{
  std::vector<string> v;
  for(int i = 0; i < nFRAMES; i++){
      stringstream sstr;
      sstr<<"/home/ljh/Plane_extraction/MyCAPE/test_component/data/pcd_cloud/"<<i<<".pcd";//我的.pcd格式数据所在目录
      v.push_back(sstr.str());
  }
  return v;
}
/*--------可视化程序，将PointXYZI转换成PointXYZRGB--------*/
boost::shared_ptr<pcl::visualization::PCLVisualizer> displayFeatures(pcl::PointCloud<pcl::PointXYZI>::ConstPtr original)
{
  //     -----Open 3D viewer and add point cloud-----
  //     --------------------------------------------
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::PointCloud<pcl::PointXYZRGB> orig;
  for(int i = 0; i < original->points.size(); i++)
  {
    pcl::PointXYZRGB point;
    point.x = original->points[i].x;
    point.y = original->points[i].y;
    point.z = original->points[i].z;

    uint32_t rgb = (static_cast<uint32_t>(100) << 16 |
                    static_cast<uint32_t>(100) << 8 | static_cast<uint32_t>(100));
    point.rgb = rgb;
    orig.points.push_back(point);
  }
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> color(orig.makeShared());
  viewer->addPointCloud<pcl::PointXYZRGB> (orig.makeShared(), color, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");

  viewer->addCoordinateSystem (1.0);//显示坐标轴,红色：x轴 绿色：y轴 蓝色：z轴
  viewer->initCameraParameters ();
  return (viewer);

}

/*--------参数初始化--------*/
void resetParameters()
{
  const int cloud_size = _vertical_scans * _horizontal_scans;
  PointType nanPoint;
  nanPoint.x = std::numeric_limits<float>::quiet_NaN();
  nanPoint.y = std::numeric_limits<float>::quiet_NaN();
  nanPoint.z = std::numeric_limits<float>::quiet_NaN();
  
  _range_mat.resize(_vertical_scans, _horizontal_scans);
  _range_mat.fill(FLT_MAX);

  _full_cloud->points.resize(cloud_size);
  //_full_info_cloud->points.resize(cloud_size);

  std::fill(_full_cloud->points.begin(), _full_cloud->points.end(), nanPoint);
  //std::fill(_full_info_cloud->points.begin(), _full_info_cloud->points.end(),nanPoint);
}

/*--------确定一帧点云数据的开始结束角度--------*/
void findStartEndAngle() {
  // start and end orientation of this cloud
  auto point = _laser_cloud_in->points.front();
  //计算角度时以x轴负轴为基准
  _seg_msg.startOrientation = -std::atan2(point.y, point.x);

  point = _laser_cloud_in->points.back();
  //最末角度为2π减去计算值
  _seg_msg.endOrientation = -std::atan2(point.y, point.x) + 2 * M_PI;

  if (_seg_msg.endOrientation - _seg_msg.startOrientation > 3 * M_PI) {
    _seg_msg.endOrientation -= 2 * M_PI;
  } else if (_seg_msg.endOrientation - _seg_msg.startOrientation < M_PI) {
    _seg_msg.endOrientation += 2 * M_PI;
  }
  _seg_msg.orientationDiff = _seg_msg.endOrientation - _seg_msg.startOrientation;
}

/*--------单帧点云数据结构化--------*/
void projectPointCloud() {
  // range image projection
  const size_t cloudSize = _laser_cloud_in->points.size();

  for (size_t i = 0; i < cloudSize; ++i) {
    PointType thisPoint = _laser_cloud_in->points[i];
    /*
    //thisPoint是激光雷达坐标系的点，x向前y向左z向上，转变成z向前x向右y向下,符合图像坐标系
    thisPoint.x = _laser_cloud_in->points[i].y;
    thisPoint.y = _laser_cloud_in->points[i].z;
    thisPoint.z = _laser_cloud_in->points[i].x;
    */
    float range = sqrt(thisPoint.x * thisPoint.x +
                      thisPoint.y * thisPoint.y +
                      thisPoint.z * thisPoint.z);

    // find the row and column index in the image for this point
    float verticalAngle = std::asin(thisPoint.z / range);
    //std::atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y));

    // 计算垂直方向上点的角度及在整个雷达点云的哪一条scan上
    int rowIdn = (verticalAngle + _ang_bottom) / _ang_resolution_Y;
    if (rowIdn < 0 || rowIdn >= _vertical_scans) {
      continue;
    }

    //水平方向上点的角度与所在线数
    float horizonAngle = std::atan2(thisPoint.x, thisPoint.y);
    int columnIdn = -round((horizonAngle - M_PI_2) / _ang_resolution_X) + _horizontal_scans * 0.5;

    if (columnIdn >= _horizontal_scans){
      columnIdn -= _horizontal_scans;
    }
    if (columnIdn < 0 || columnIdn >= _horizontal_scans){
      continue;
    }
    if (range < 0.1){
      continue;
    }
    //在rangeMat矩阵中保存该点的深度，单位：m
    _range_mat(rowIdn, columnIdn) = range;
    thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;
    size_t index = columnIdn + rowIdn * _horizontal_scans;
    //index是计算的列数+计算的行数×固定列数
    //完成了将点分段存储在一个一维数组中
    _full_cloud->points[index] = thisPoint;
    //_full_info_cloud则存储点云的xyz坐标以及range
    //_full_info_cloud->points[index] = thisPoint;
    //_full_info_cloud->points[index].intensity = range;

    int cell_r = rowIdn/PATCH_SIZE;
    int local_r = rowIdn%PATCH_SIZE;
    int cell_c = columnIdn/PATCH_SIZE;
    int local_c = columnIdn%PATCH_SIZE;
    int num = (cell_r*nr_horizontal_cells+cell_c)*PATCH_SIZE*PATCH_SIZE + local_r*PATCH_SIZE + local_c;
    cloud_array_organized(num,0) = thisPoint.y;
    cloud_array_organized(num,1) = thisPoint.z;        
    cloud_array_organized(num,2) = thisPoint.x;
    //cloud_array_organized是一个nx3矩阵，0-PATCH_SIZE×PATCH_SIZE是第一个cell内的点，依次类推
  }
}

int main(int argc, char** argv)
{
  vector<string> files = loadData();
  for(int i=0; i<nFRAMES; i++)
  {
    //pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);
    if (pcl::io::loadPCDFile<pcl::PointXYZI> (files[i], *_laser_cloud_in) == -1){
        PCL_ERROR ("Couldn't read file \n");
        return (-1);
    }
    //cout<<_laser_cloud_in->points.size()<<endl;
    resetParameters();
    //findStartEndAngle();
    projectPointCloud();

    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*_full_cloud,*_full_cloud, indices); 

    // Initialize CAPE
    plane_detector = new CAPE(_vertical_scans, _horizontal_scans, PATCH_SIZE, PATCH_SIZE, cylinder_detection, COS_ANGLE_MAX, MAX_MERGE_DIST);
    //cout<<_full_cloud->points.size()<<endl;
    //cv::Mat_<cv::Vec3b> seg_rz = cv::Mat_<cv::Vec3b>(_vertical_scans,_horizontal_scans,cv::Vec3b(0,0,0));
    cv::Mat_<uchar> seg_output = cv::Mat_<uchar>(_vertical_scans,_horizontal_scans,uchar(0));

    // Run CAPE
    int nr_planes, nr_cylinders;
    vector<PlaneSeg> plane_params;
    vector<CylinderSeg> cylinder_params;
    double t1 = cv::getTickCount();
    plane_detector->process(cloud_array_organized, nr_planes, nr_cylinders, seg_output, plane_params, cylinder_params);
    double t2 = cv::getTickCount();
    double time_elapsed = (t2-t1)/(double)cv::getTickFrequency();
    cout<<"Total time elapsed: "<<time_elapsed<<endl;//完成平面和圆柱特征提取花费的时间

    //预先配置100种配色，前50种给平面特征使用，后50种给圆柱特征使用
    std::vector<cv::Vec3b> color_code;//存放100不同的颜色，前50用于平面显示
    for(int i=0; i<100;i++){
      cv::Vec3b color;
      color[0]=rand()%255;
      color[1]=rand()%255;
      color[2]=rand()%255;
      color_code.push_back(color);
    }
    // 为平面添加特定颜色
    color_code[0][0] = 0; color_code[0][1] = 0; color_code[0][2] = 255;
    color_code[1][0] = 255; color_code[1][1] = 0; color_code[1][2] = 204;
    color_code[2][0] = 255; color_code[2][1] = 100; color_code[2][2] = 0;
    color_code[3][0] = 0; color_code[3][1] = 153; color_code[3][2] = 255;
    // 为圆柱体添加特定颜色
    color_code[50][0] = 178; color_code[50][1] = 255; color_code[50][2] = 0;
    color_code[51][0] = 255; color_code[51][1] = 0; color_code[51][2] = 51;
    color_code[52][0] = 0; color_code[52][1] = 255; color_code[52][2] = 51;

    //使用彩色编码映射分割点云
    pcl::PointCloud<pcl::PointXYZRGB> seg_cloud_show;
    uchar * sCode;
    int code;
    for(int r=0; r<_vertical_scans; r++){
      sCode = seg_output.ptr<uchar>(r);
      for(int c=0; c<_horizontal_scans; c++){
        code = *sCode;
        pcl::PointXYZRGB point;
        int i = r*_horizontal_scans + c;
        point.x = _full_cloud->points[i].x;
        point.y = _full_cloud->points[i].y;
        point.z = _full_cloud->points[i].z;
        if(code > 0){
          point.r = color_code[code-1][0];
          point.g = color_code[code-1][1];
          point.b = color_code[code-1][2];
        }
        else{
          point.r = int(255);
          point.g = int(255);
          point.b = int(255);
        }
        seg_cloud_show.points.push_back(point);
        sCode++;
      }
    }
   
    boost::shared_ptr< pcl::visualization::PCLVisualizer > viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    //设置点云颜色
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> color(seg_cloud_show.makeShared());
    viewer->addPointCloud<pcl::PointXYZRGB>(seg_cloud_show.makeShared(), color, "sample_cloud");
    //viewer->addPointCloud<pcl::PointXYZI>(_full_cloud,"sample_cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample_cloud");
    viewer->addCoordinateSystem (1.0);//显示坐标轴,红色：x轴 绿色：y轴 蓝色：z轴
    viewer->spin();
    /*
    读取到文件，则对点云指针的取值 *cloud 进行操作
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
    viewer = displayFeatures(_full_cloud);

    while(!viewer->wasStopped()){
        viewer->spinOnce(100);//可视化100ms
    }
    */
  }
  return 0;
}