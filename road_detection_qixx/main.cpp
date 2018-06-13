#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string.h>
#include <string>
#include <stdio.h>
#include <time.h>

using namespace cv;
using namespace std;

//　函数声明
Mat readImage(string path);
Mat darkChannelInfo(Mat image);
Mat KMeansCluster(Mat dark);
Mat edgeExtract(Mat image);
//　中值滤波 3x3
Mat MedianFiltering(const Mat &src);
uchar Median(uchar n1, uchar n2, uchar n3, uchar n4, uchar n5, uchar n6, uchar n7, uchar n8, uchar n9);
Mat medianFiltering(Mat image, int dim);
// 均值滤波 3x3
Mat AverFiltering(const Mat &src);
Mat averFiltering(Mat image, int dim);
// 高斯滤波
Mat gaussianFilter(Mat image, int dim);
// 去除连通区域
Mat removeSmallRegion(Mat& src, int AreaLimit, int CheckMode, int NeihborMode);
// 检测所有直线
void lineDetect(Mat& image, vector<Point>& key_point, int n);
// 随机选择点进行曲线拟合
vector<Point> randomSelectUp(Mat image);
vector<Point> randomSelectDown(Mat image);
//　绘制道路三角形
void roadTrangle(Mat& image, double i);
// 
Mat lineExtract(Mat img, double i, double u);

void softVoting(Mat& img);
// 主函数
int main()
{
    /* 读取图像 */
    string path = "0524_15/phone/4.jpg";
    Mat image = readImage(path);
    imwrite("0.jpg", image);
    /* 图像相关信息 */
    int channels = image.channels();    // 3
    int Rows = image.rows;              // 1920
    int Cols = image.cols;              // 2560 
    /* 暗通道处理　*/
    Mat darkChannel = darkChannelInfo(image);
    imwrite("1.jpg", darkChannel);
    /* 高斯滤波　*/
    Mat afterFilter = gaussianFilter(darkChannel, 3);
    /* KMeans聚类　*/
    Mat KMeans = KMeansCluster(afterFilter);
    imwrite("2.jpg", KMeans);
    /* 提取图像边缘　*/
    Mat edge = edgeExtract(KMeans);
    imwrite("3.jpg", edge);
    Mat res0 = lineExtract(edge, 0.6, 0.05);
    imwrite("7.jpg", res0);
    /* 去除小连通区域　*/
    Mat res = removeSmallRegion(edge, 2000, 0, 0);
    res = removeSmallRegion(res, 2000, 1, 0);
    imwrite("4.jpg", res);
    /* 分上下两部分选择图中的点拟合两条曲线 */
    vector<Point> pointsUP = randomSelectUp(res);
    vector<Point> pointsDown = randomSelectDown(res);
    lineDetect(image, pointsUP, 3);
    lineDetect(image, pointsDown, 3);
    imwrite("5.jpg", image);
    /* 绘制道路三角形　*/
    roadTrangle(image, 0.55);
    imwrite("6.jpg", image);

    /* 求两条曲线的交点确定为道路消失点 */


    return 0;
}

/* 从二值图中随机选取点　*/
vector<Point> randomSelectUp(Mat image)
{
    vector<Point> points;
    int row = image.rows;
    int col = image.cols;
    int channel = image.channels();
    for(int i = 0; i < row/2; i++)
    {
        for(int j = 0; j < col; j++)
        {
            if(image.at<uchar>(i, j) != 0)
            {
                Point tmp;
                tmp.x = j;
                tmp.y = i;
                points.push_back(tmp);
            }
        }
    }
    return points;
}
vector<Point> randomSelectDown(Mat image)
{
    vector<Point> points;
    int row = image.rows;
    int col = image.cols;
    int channel = image.channels();
    for(int i = row / 2 + 1; i < row; i++)
    {
        for(int j = 0; j < col; j++)
        {
            if(image.at<uchar>(i, j) != 0)
            {
                Point tmp;
                tmp.x = j;
                tmp.y = i;
                points.push_back(tmp);
            }
        }
    }
    return points;
}

/* 读取图像　*/
Mat readImage(string path)
{
    Mat image = imread(path);
	if(!image.data)		
	{
		cout << "Fail to load image!!" << endl;
		system("pause");
	}
    return image;
}

/* 得到暗通道图像　*/
Mat darkChannelInfo(Mat image)  
{
    int row = image.rows;
    int col = image.cols;
    Mat minRGB = Mat::zeros(row, col,  CV_32FC1);  
    Mat dark = Mat::zeros(row, col,  CV_32FC1);  
    
    /*　选择三个通道中值最小的通道组成一副新的图片　*/ 
    Vec3b temp;  
    for(int m = 0; m < row; m++)  
    {  
        for(int n = 0; n < col; n++)  
        { 
            temp = image.at<Vec3b>(m, n);  
            minRGB.at<float>(m, n) = min(min(temp.val[0], temp.val[1]), temp.val[2]);  
        }  
    }   
    /* 最小值滤波
     *    最大(小)值滤波是指在图像中以当前像素 f(i,j)为中心切出一个 N*M(例如 3*3)像素组成的图像块
     *    设当前像素 f(i,j)的灰度值为 g(i,j)时,则 g(i,j)取 N*N 个诸像素灰度值中的最大(小)值.
     */
    int scale = 3;  
    Mat border;                                                                         // 边界扩展后矩阵
    int radius = (scale - 1) / 2;                                                       // 半径
    copyMakeBorder(minRGB, border, radius, radius, radius, radius, BORDER_REPLICATE);   //　扩展边界
    for (int i = 0; i < col; i++)  
    {  
        for (int j = 0; j < row; j++)  
        {  
            Mat roi;
            roi = border(Rect(i, j, scale, scale));
            double minVal = 0;
            double maxVal = 0;
            Point minLoc = Point(0); 
            Point maxLoc = Point(0);
            minMaxLoc(roi, &minVal, &maxVal, &minLoc, &maxLoc, noArray());  
            dark.at<float>(Point(i, j)) = (float)minVal;                               //　求得ROI中最小值
        }  
    }  
    return dark; 
}  

/* KMeans对暗通道图像进行聚类 */
Mat KMeansCluster(Mat dark)
{
    int row = dark.rows;
    int col = dark.cols;
    int clusterCount = 3;
    Mat data, labels;
    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < col; j++)
        {
            Vec3d point = dark.at<float>(i, j);
            Mat tmp = (Mat_<float>(1, 1) << (float)point[0]);
            data.push_back(tmp); 
        }
    }
	kmeans(data, clusterCount, labels, 
        TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0),    //用最大迭代次数或者精度作为迭代条件，看哪个条件先满足
		3,                                                            // 聚类多次
		KMEANS_PP_CENTERS);             
    int index = 0;
    Mat res(row, col, CV_8UC3);
    for(int i = 0; i < row; i++) 
    { 
        for(int j = 0; j < col; j++) 
        { 
            index = i * col + j; 
            int label = labels.at<int>(index); 
            if(label == 1) 
            { 
                res.at<Vec3b>(i, j)[0] = 0; 
                res.at<Vec3b>(i, j)[1] = 0; 
                res.at<Vec3b>(i, j)[2] = 0; 
            }
            else if(label == 2) 
            { 
                res.at<Vec3b>(i, j)[0] = 0; 
                res.at<Vec3b>(i, j)[1] = 0; 
                res.at<Vec3b>(i, j)[2] = 0; 
            }
            else
            { 
                res.at<Vec3b>(i, j)[0] = 255; 
                res.at<Vec3b>(i, j)[1] = 255; 
                res.at<Vec3b>(i, j)[2] = 255; 
            } 
        } 
    } 
    return res;
}

// 求以当前元素为中心的3x3矩阵的中值
uchar Median(uchar n1, uchar n2, uchar n3, uchar n4, uchar n5, uchar n6, uchar n7, uchar n8, uchar n9) 
{  
    uchar arr[9];  
    arr[0] = n1;  
    arr[1] = n2;  
    arr[2] = n3;  
    arr[3] = n4;  
    arr[4] = n5;  
    arr[5] = n6;  
    arr[6] = n7;  
    arr[7] = n8;  
    arr[8] = n9;  
    for (int gap = 9 / 2; gap > 0; gap /= 2)
    {
        for (int i = gap; i < 9; i++)
        {
            for (int j = i - gap; j >= 0 && arr[j] > arr[j + gap]; j -= gap)
            {
                swap(arr[j], arr[j + gap]);
            }
        }
    }  
    return arr[4];
} 

/* 中值滤波，去掉椒盐噪声　*/
Mat MedianFiltering(const Mat &src)
{  
    if(!src.data)
    {
		cout << "Fail to load image!!" << endl;
		system("pause");
    }

    Mat dst(src.size(), src.type());  
    for(int i = 0; i < src.rows; i++)  
        for(int j = 0; j < src.cols; j++)
        {  
            if((i - 1) > 0 && (i + 1) < src.rows && (j - 1) > 0 && (j + 1) < src.cols)
            {  
                dst.at<Vec3b>(i, j)[0] = Median(src.at<Vec3b>(i, j)[0], src.at<Vec3b>(i + 1, j + 1)[0],  
                    src.at<Vec3b>(i + 1, j)[0], src.at<Vec3b>(i, j + 1)[0], src.at<Vec3b>(i + 1, j - 1)[0],  
                    src.at<Vec3b>(i - 1, j + 1)[0], src.at<Vec3b>(i - 1, j)[0], src.at<Vec3b>(i, j - 1)[0],  
                    src.at<Vec3b>(i - 1, j - 1)[0]);  
                dst.at<Vec3b>(i, j)[1] = Median(src.at<Vec3b>(i, j)[1], src.at<Vec3b>(i + 1, j + 1)[1],  
                    src.at<Vec3b>(i + 1, j)[1], src.at<Vec3b>(i, j + 1)[1], src.at<Vec3b>(i + 1, j - 1)[1],  
                    src.at<Vec3b>(i - 1, j + 1)[1], src.at<Vec3b>(i - 1, j)[1], src.at<Vec3b>(i, j - 1)[1],  
                    src.at<Vec3b>(i - 1, j - 1)[1]);  
                dst.at<Vec3b>(i, j)[2] = Median(src.at<Vec3b>(i, j)[2], src.at<Vec3b>(i + 1, j + 1)[2],  
                    src.at<Vec3b>(i + 1, j)[2], src.at<Vec3b>(i, j + 1)[2], src.at<Vec3b>(i + 1, j - 1)[2],  
                    src.at<Vec3b>(i - 1, j + 1)[2], src.at<Vec3b>(i - 1, j)[2], src.at<Vec3b>(i, j - 1)[2],  
                    src.at<Vec3b>(i - 1, j - 1)[2]);  
            }  
            else
            {
                dst.at<Vec3b>(i, j) = src.at<Vec3b>(i, j); 
            } 
        }  
    return dst;
}  

Mat AverFiltering(const Mat &src)
{
    if(!src.data)
    {
		cout << "Fail to load image!!" << endl;
		system("pause");
    }
    Mat dst(src.size(), src.type());  
    for (int i = 1; i < src.rows; i++)
    {
        for (int j = 1; j < src.cols; j++)
        {  
            if ((i - 1 >= 0) && (j - 1) >= 0 && (i + 1) < src.rows && (j + 1) < src.cols) 
            {
                dst.at<Vec3b>(i, j)[0] = (src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i - 1, j - 1)[0] + src.at<Vec3b>(i - 1, j)[0] + src.at<Vec3b>(i, j - 1)[0] +  
                    src.at<Vec3b>(i - 1, j + 1)[0] + src.at<Vec3b>(i + 1, j - 1)[0] + src.at<Vec3b>(i + 1, j + 1)[0] + src.at<Vec3b>(i, j + 1)[0] +  
                    src.at<Vec3b>(i + 1, j)[0]) / 9;  
                dst.at<Vec3b>(i, j)[1] = (src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i - 1, j - 1)[1] + src.at<Vec3b>(i - 1, j)[1] + src.at<Vec3b>(i, j - 1)[1] +  
                    src.at<Vec3b>(i - 1, j + 1)[1] + src.at<Vec3b>(i + 1, j - 1)[1] + src.at<Vec3b>(i + 1, j + 1)[1] + src.at<Vec3b>(i, j + 1)[1] +  
                    src.at<Vec3b>(i + 1, j)[1]) / 9;  
                dst.at<Vec3b>(i, j)[2] = (src.at<Vec3b>(i, j)[2] + src.at<Vec3b>(i - 1, j - 1)[2] + src.at<Vec3b>(i - 1, j)[2] + src.at<Vec3b>(i, j - 1)[2] +  
                    src.at<Vec3b>(i - 1, j + 1)[2] + src.at<Vec3b>(i + 1, j - 1)[2] + src.at<Vec3b>(i + 1, j + 1)[2] + src.at<Vec3b>(i, j + 1)[2] +  
                    src.at<Vec3b>(i + 1, j)[2]) / 9;  
            }  
            else
            {
                dst.at<Vec3b>(i, j) = src.at<Vec3b>(i, j);  
            }  
        }
    }  
    return dst;
}  

Mat medianFiltering(Mat image, int dim)
{
    Mat result;
    medianBlur(image, result, dim);
    return result;
}

Mat averFiltering(Mat image, int dim)
{
    Mat result;
    blur(image, result, Size(dim, dim), Point(-1, -1));
    return result;
}

Mat gaussianFilter(Mat image, int dim)
{
    Mat result;
    GaussianBlur(image, result, Size(dim, dim), 1, 1);
    return result;
}

Mat edgeExtract(Mat image)
{
    Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3), Point(-1, -1));
    /* 边缘提取方法*/
    Mat edge, result;
    /* 开运算:先腐蚀后膨胀
     * 闭运算:先膨胀后腐蚀
     */
    morphologyEx(image, result, MORPH_CLOSE, element);
    Canny(result, edge, 50, 150, 3);
    
    /* 去掉小连通分量　或者　拟合边缘线　*/
    vector<vector<Point> >contours;
    vector<Vec4i> hierarchy;
    // 轮廓检测
    findContours(edge, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
    // 绘制轮廓
    Mat res(image.size(), CV_8UC1, Scalar(0));
    drawContours(res, contours, -1, Scalar(255), 3);
    return res;
}

/* 去除小连通区域
 * 新建一幅标签图像初始化为0像素点
 * 为了记录每个像素点检验状态的标签，0代表未检查，1代表正在检查,2代表检查不合格（需要反转颜色），3代表检查合格或不需检查 
 * CheckMode: 0代表去除黑区域，1代表去除白区域; 
 * NeihborMode：0代表4邻域，1代表8邻域;  
 */
Mat removeSmallRegion(Mat& src, int AreaLimit, int CheckMode, int NeihborMode)  
{
    Mat dst(src.size(), CV_8UC1, Scalar(0));
    int RemoveCount = 0;       //记录除去的个数  
    Mat Pointlabel = Mat::zeros(src.size(), CV_8UC1);  

    if(CheckMode == 1)  
    {  
        cout << "Mode: 去除小区域. " << endl;  
        for(int i = 0; i < src.rows; i++)    
        {    
            uchar* iData = src.ptr<uchar>(i);  
            uchar* iLabel = Pointlabel.ptr<uchar>(i);  
            for(int j = 0; j < src.cols; j++)    
            {    
                if (iData[j] < 10)    
                {    
                    iLabel[j] = 3;   
                }    
            }    
        }
    }
    else  
    {  
        cout << "Mode: 去除孔洞. " << endl;  
        for(int i = 0; i < src.rows; i++)    
        {    
            uchar* iData = src.ptr<uchar>(i);  
            uchar* iLabel = Pointlabel.ptr<uchar>(i);  
            for(int j = 0; j < src.cols; j++)    
            {    
                if (iData[j] > 10)    
                {    
                    iLabel[j] = 3;   
                }    
            }    
        }    
    }  
  
    vector<Point2i> NeihborPos;  //记录邻域点位置  
    NeihborPos.push_back(Point2i(-1, 0));  
    NeihborPos.push_back(Point2i(1, 0));  
    NeihborPos.push_back(Point2i(0, -1));  
    NeihborPos.push_back(Point2i(0, 1));  
    if(NeihborMode == 1)  
    {  
        cout << "Neighbor mode: 8邻域." << endl;  
        NeihborPos.push_back(Point2i(-1, -1));  
        NeihborPos.push_back(Point2i(-1, 1));  
        NeihborPos.push_back(Point2i(1, -1));  
        NeihborPos.push_back(Point2i(1, 1));  
    }  
    else 
    {
        cout << "Neighbor mode: 4邻域." << endl; 
    }

    int NeihborCount = 4 + 4 * NeihborMode;  
    int CurrX = 0, CurrY = 0;  
    
    for(int i = 0; i < src.rows; i++)    
    {    
        uchar* iLabel = Pointlabel.ptr<uchar>(i);  
        for(int j = 0; j < src.cols; j++)    
        {    
            if (iLabel[j] == 0)    
            {    
                //********开始该点处的检查**********  
                vector<Point2i> GrowBuffer;                                      //堆栈，用于存储生长点  
                GrowBuffer.push_back( Point2i(j, i) );  
                Pointlabel.at<uchar>(i, j)=1;  
                int CheckResult=0;                                               //用于判断结果（是否超出大小），0为未超出，1为超出  
  
                for(int z = 0; z < GrowBuffer.size(); z++)  
                {
                    for(int q = 0; q < NeihborCount; q++)                                      //检查四个邻域点  
                    {  
                        CurrX = GrowBuffer.at(z).x + NeihborPos.at(q).x;  
                        CurrY = GrowBuffer.at(z).y + NeihborPos.at(q).y;  
                        if(CurrX >= 0 && CurrX < src.cols && CurrY >= 0 && CurrY < src.rows)  //防止越界  
                        {  
                            if(Pointlabel.at<uchar>(CurrY, CurrX) == 0)  
                            {  
                                GrowBuffer.push_back(Point2i(CurrX, CurrY));  //邻域点加入buffer  
                                Pointlabel.at<uchar>(CurrY, CurrX) = 1;           //更新邻域点的检查标签，避免重复检查  
                            }  
                        }  
                    }
                }  
                if(GrowBuffer.size() > AreaLimit)
                {
                    CheckResult = 2;                 //判断结果（是否超出限定的大小），1为未超出，2为超出
                }
                else
                {
                    CheckResult=1;   
                    RemoveCount++;
                }  
                for(int z = 0; z < GrowBuffer.size(); z++)                         //更新Label记录  
                {  
                    CurrX = GrowBuffer.at(z).x;   
                    CurrY = GrowBuffer.at(z).y;  
                    Pointlabel.at<uchar>(CurrY, CurrX) += CheckResult;  
                }
            }    
        }    
    }    
  
    CheckMode = 255 * (1 - CheckMode);  
    //开始反转面积过小的区域  
    for(int i = 0; i < src.rows; i++)    
    {    
        uchar* iData = src.ptr<uchar>(i);  
        uchar* iDstData = dst.ptr<uchar>(i);  
        uchar* iLabel = Pointlabel.ptr<uchar>(i);  
        for(int j = 0; j < src.cols; j++)    
        {    
            if (iLabel[j] == 2)    
            {    
                iDstData[j] = CheckMode;   
            }    
            else if(iLabel[j] == 3)  
            {  
                iDstData[j] = iData[j];  
            }  
        }    
    }   
      
    cout << RemoveCount << " objects removed." << endl;  
    return dst;
}  
/*
 * 最小二乘法进行曲线拟合
 */
void lineDetect(Mat& image, vector<Point>& key_point, int n)
{
    int N = key_point.size();
    Mat X = Mat::zeros(n + 1, n +1, CV_64FC1);
    /* 构造矩阵X */
    for(int i = 0; i < n + 1; i++)
    {
        for(int j = 0; j < n + 1; j++)
        {
            for(int k = 0; k < N; k++)
            {
                X.at<double>(i, j) = X.at<double>(i, j) + pow(key_point[k].x, i + j);
            }
        }
    }
    /* 构造矩阵Y */
    Mat Y = Mat::zeros(n + 1, 1, CV_64FC1);
    for(int i = 0; i < n + 1; i++)
    {
        for(int j = 0; j < N; j++)
        {
            Y.at<double>(i, 0) = Y.at<double>(i, 0) + pow(key_point[j].x, i) * key_point[j].y;
        }
    }
    /* 利用solve求解方重组得到矩阵A */
    Mat A = Mat::zeros(n + 1, 1, CV_64FC1);
    solve(X, Y, A, DECOMP_LU);
    /* 表示为一元多次曲线　*/
    vector<Point> point_fitted;
    for(int x = 0; x < key_point.size(); x++)
    {
        double y = A.at<double>(0, 0) + A.at<double>(1, 0) * x + A.at<double>(2, 0) * pow(x, 2) + A.at<double>(3, 0) * pow(x, 3);
        point_fitted.push_back(Point(x, y));
    }
    polylines(image, point_fitted, false, Scalar(0, 255, 0));
}

/* 画道路三角形　*/
void roadTrangle(Mat& image, double i)
{
    Point a = Point(0, image.rows);
    Point b = Point(image.cols / 2, i * image.rows);
    Point c = Point(image.cols, image.rows);
    line(image, a, b, Scalar(255), 2);
    line(image, b, c, Scalar(255), 2);
    line(image, a, c, Scalar(255), 2);
}

Mat lineExtract(Mat img, double i, double u)
{
    Mat res = Mat::zeros(img.rows, img.cols, CV_8UC1);
    /* 检测图像中的垂直直线　*/
    vector<Vec4i> lines;
    HoughLinesP(img, lines, 1, CV_PI / 180, 50, 50, 10);
    for(size_t i = 0; i < lines.size(); i++)
    {
        Vec4i l = lines[i];
        Point p1 = Point(l[0], l[1]);
        Point p2 = Point(l[2], l[3]); 
        float len = sqrtf((p2.x - p1.x) * (p2.x - p1.x) + (p2.y - p1.y) * (p2.y - p1.y));
        float angle = abs(atan2(p1.y - p2.y, p1.x - p2.x) * 180 / CV_PI);
        if(angle >= 80 && angle <= 100)
        {
            line(res, p1, p2, Scalar(1), 2);
        }
    }
    vector<Point> maxH, minH;
    /* 垂直包络 */
    for(int j = 0; j < img.cols; j += u *img.cols)
    {
        Point maxTmp = Point(img.cols, img.rows);
        Point minTmp = Point(0, 0);
        for(int i = 0; i < u * img.cols; i++)
        {
            for(int k = 0; k < img.rows; k++)
            {
                if(res.at<uchar>(k, j + i) != 0 && k < maxTmp.y)
                {
                    maxTmp.y = k;
                    maxTmp.x = j + i;
                }
                if(res.at<uchar>(k, j + i) != 0 && k > minTmp.y)
                {
                    minTmp.y = k;
                    minTmp.x = j + i;
                }
            }
        }
        if(maxTmp.x != 0 && maxTmp.y != 0 && maxTmp.x != img.cols && maxTmp.y != img.rows)
        {
            maxH.push_back(maxTmp);
        }
        if(minTmp.x != 0 && minTmp.y != 0 && minTmp.x != img.cols && minTmp.y != img.rows)
        {
            minH.push_back(minTmp);
        }
    }
    polylines(res, maxH, false, Scalar(255), 4);
    polylines(res, minH, false, Scalar(255), 4);
    /* 求另一部分区域　*/
    for(size_t i = 0; i < lines.size(); i++)
    {
        Vec4i l = lines[i];
        Point p1 = Point(l[0], l[1]);
        Point p2 = Point(l[2], l[3]); 
        float len = sqrtf((p2.x - p1.x) * (p2.x - p1.x) + (p2.y - p1.y) * (p2.y - p1.y));
        float angle = abs(atan2(p1.y - p2.y, p1.x - p2.x) * 180 / CV_PI);
        if(len > 120 && angle >= 5 && angle <= 90 || len > 120 && angle >= 90 && angle <= 175)
        {
            line(res, p1, p2, Scalar(1), 2);
        }
    }
    vector<Point> maxHG, minHG;
    for(int j = 0; j < img.cols; j += u *img.cols)
    {
        Point maxTmp = Point(img.cols, img.rows);
        Point minTmp = Point(0, 0);
        for(int i = 0; i < u * img.cols; i++)
        {
            for(int k = 0; k < img.rows; k++)
            {
                if(res.at<uchar>(k, j + i) != 0 && k < maxTmp.y)
                {
                    maxTmp.y = k;
                    maxTmp.x = j + i;
                }
                if(res.at<uchar>(k, j + i) != 0 && k > minTmp.y)
                {
                    minTmp.y = k;
                    minTmp.x = j + i;
                }
            }
        }
        if(maxTmp.x != 0 && maxTmp.y != 0 && maxTmp.x != img.cols && maxTmp.y != img.rows)
        {
            maxHG.push_back(maxTmp);
        }
        if(minTmp.x != 0 && minTmp.y != 0 && minTmp.x != img.cols && minTmp.y != img.rows)
        {
            minHG.push_back(minTmp);
        }
    }
    polylines(res, maxHG, false, Scalar(255));
    polylines(res, minHG, false, Scalar(255));
    /* 画道路三角形并且判断点是否位于三角形内　*/
    Point a = Point(0, img.rows);
    Point b = Point(img.cols / 2, i * img.rows);
    Point c = Point(img.cols, img.rows);
    line(res, a, b, Scalar(255), 2);
    line(res, b, c, Scalar(255), 2);
    line(res, a, c, Scalar(255), 2);
    //此处应该添加判断
    Point vanishPoint = Point(img.cols, img.rows);
    for(int i = 0; i < minH.size(); i++)
    {
        for(int j = 0; j < minHG.size(); j++)
        {
            if(abs(minH[i].x - minHG[j].x) < 3 && abs(minH[i].y - minHG[j].y) < 3)
            {
                if(vanishPoint.y > minH[i].y)
                {
                    vanishPoint.x = (minH[i].x + minHG[j].x) / 2;
                    vanishPoint.y = (minH[i].y + minHG[j].y) / 2; 
                }
            }
        }
    }
    circle(res, vanishPoint, 10, Scalar(255), 5);
    /* 轮廓检测 */
    vector<vector<Point> >contours;
    vector<Vec4i> hierarchy;
    findContours(res, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
    return res;
}

void softVoting(Mat& img)
{
    /* bottom voting */
    /* area voting */
    /* triangle voting */
}
