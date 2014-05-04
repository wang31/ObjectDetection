/*
 * image.cpp
 *
 *  Created on: Apr 29, 2014
 *      Author: zhengwang
 */


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <cstring>
using namespace cv;
using namespace std;


/// Global Variables
Mat src; Mat hsv; Mat hue;Mat mask;
Mat src_no_alpha;
Mat candidate; Mat hsv_candidate; Mat hue_candidate;
int bins = 35;
int w=0;
int ct1 = -1;
int ct2 = -1;
int search_x=120,search_y=120;
int lo = 20; int up = 20;

struct locf{
	int sum,x,y;
	locf(int s,int a,int b){ sum=s;x=a,y=b;}
};


struct prof{
	int xl,yl,bv;
};


locf biggest(0,0,0);


int calcBrightness(Mat m){
	unsigned int sum=0;
	 unsigned int cur;

	  for(int i = 0; i < m.rows; i++)
	  {
	      const unsigned char* Mi = m.ptr<unsigned char>(i);

	      for(int j =0; j <m.cols; j++){
	    	  cur=Mi[j];
	    	  if(cur<160)
	    		  cur=0;
	    	  sum+=cur;
	    	  //cout<<cur<<" ";
	      }
	     // cout<<endl;
	  }

	return sum;
}
prof p[12];
void Hist_and_Backproj2( )
{
  p[0].yl=80;p[0].xl=140;p[0].bv=35;
  p[1].yl=60;p[1].xl=70;p[1].bv=35;
  p[2].yl=60;p[2].xl=70;p[2].bv=35;
  p[3].yl=90;p[3].xl=125;p[3].bv=35;
  p[4].yl=70;p[4].xl=90;p[4].bv=35;
  p[5].yl=55;p[5].xl=75;p[5].bv=33;
  p[6].yl=120;p[6].xl=100;p[6].bv=34;
  p[7].yl=120;p[7].xl=100;p[7].bv=34;
  p[8].yl=50;p[8].xl=50;p[8].bv=35;
  p[9].yl=135;p[9].xl=50;p[9].bv=34;
  p[10].yl=40;p[10].xl=50;p[10].bv=34;
  p[11].yl=60;p[11].xl=70;p[11].bv=34;
}

/**
 * @function Hist_and_Backproj
 * @brief Callback to Trackbar
 */
void Hist_and_Backproj(int, void* )
{
  MatND hist;
  int histSize = MAX( bins, 2 );
  float hue_range[] = { 1, 359 };
  const float* ranges = { hue_range };

  /// Get the Histogram and normalize it
  calcHist( &hue, 1, 0, Mat(), hist, 1, &histSize, &ranges, true, false );
  normalize( hist, hist, 0, 255, NORM_MINMAX, -1, Mat() );

  /// Get Backprojection
  Mat backproj;
  calcBackProject( &hue_candidate, 1, 0, hist, backproj, &ranges, 1, true );
  /// Draw the backproj
  cout<<bins<<endl;
  imshow( "BackProj", backproj );

  double thold=0.15;


  unsigned int sum=0;
  int maxval=search_x*search_y*255;
  Mat B,C;

  for(int i = 25; i < backproj.rows-search_y; i++)
  {
      const unsigned char* Mi = backproj.ptr<unsigned char>(i);
      B = backproj( Range(i, i+search_y),Range::all());
      for(int j =0; j <backproj.cols-search_x; j++){
    	  C= B(Range::all() , Range(j,j+search_x));
    	  sum=calcBrightness(C);
    	  if(sum>biggest.sum){
    		  biggest.sum=sum;
    		  biggest.x=j;
    		  biggest.y=i;
    	  }
        // cout<<ui<<" ";
      }
     // cout<<endl;
  }
  cout<<"I find the biggest stuff ::::   on earth :::  "<<biggest.x<<" "<<biggest.y<<endl;

  /// Draw the histogram
  int w = 400; int h = 400;
  int bin_w = cvRound( (double) w / histSize );
  Mat histImg = Mat::zeros( w, h, CV_8UC3 );

  for( int i = 0; i < bins; i ++ )
     { rectangle( histImg, Point( i*bin_w, h ), Point( (i+1)*bin_w, h - cvRound( hist.at<float>(i)*h/255.0 ) ), Scalar( 0, 0, 255 ), -1 ); }

  imshow( "Histogram", histImg );
}
int findKey(string s){
	if(s.find("orange_lev0")!=string::npos){
		return 0;
	}else if(s.find("orange_lev1-1")!=string::npos){
		return 1;
	}else if(s.find("orange_lev2-1")!=string::npos){
		return 2;
	}else if(s.find("us_flag_lev0")!=string::npos){
		return 4;
	}else if(s.find("us_flag_lev1-1")!=string::npos){
		return 5;
	}else if(s.find("us_flag_lev2-1")!=string::npos){
		return 5;
	}
	else if(s.find("starbucks_lev0")!=string::npos){
			return 6;
	}
	else if(s.find("starbucks_lev1-1")!=string::npos){
			return 7;
	}
	else if(s.find("starbucks_lev2-1")!=string::npos){
			return 8;
	}
	else if(s.find("coca_lev1-1")!=string::npos){
			return 9;
	}else if(s.find("coca_lev1-1")!=string::npos){
		return 10;
	}
	else {
			return 11;
	}
}
void pickPoint (char * point )
{
  int l=0;
  string s(point);
  w=findKey(s);
}

// generate a .ppm file and use that file.
Mat getImage(char* filepath, char* alphafile, int mode, bool method){
	  FILE* IN_FILE; // fb.raw
	  FILE* ALPHA_FILE = NULL;
	  FILE* OUT_FILE; // fb.ppm
	  unsigned char red, green, blue; // 8-bits each
	  unsigned int maxval; // max color val
	  unsigned short Width, Height;
	  size_t i;

	  IN_FILE = fopen(filepath, "rb");
	  OUT_FILE = fopen("source.ppm", "wb");
	  if(alphafile != NULL)
		  ALPHA_FILE = fopen(alphafile, "rb");

	  Width = 352;
	  Height = 288;
	  maxval = 255;

	  // P3 - PPM "plain" header
	  //fprintf(outfile, "P3\n#created with rgb2ppm\n%d %d\n%d\n", width, height, maxval);


	  char *Rbuf = new char[Height*Width];
	  char *Gbuf = new char[Height*Width];
	  char *Bbuf = new char[Height*Width];

	  char *Abuf = NULL;
		if(ALPHA_FILE != NULL){
			Abuf = new char[Height*Width];
			for(i = 0; i < Width*Height; i++){
				Abuf[i] = fgetc(ALPHA_FILE);
			}
			fclose(ALPHA_FILE);
		}
	  	for (i = 0; i < Width*Height; i ++)
		{
			char tmp = fgetc(IN_FILE);
	  		if(Abuf != NULL && Abuf[i] == 0x00){
	  			Rbuf[i] = 0x00;
			}
			else Rbuf[i] = tmp;
		}
		for (i = 0; i < Width*Height; i ++)
		{
			char tmp = fgetc(IN_FILE);
			if(Abuf != NULL && Abuf[i] == 0x00){
				  Gbuf[i] = 0x00;
			}
			else Gbuf[i] = tmp;
		}
		for (i = 0; i < Width*Height; i ++)
		{
			char tmp = fgetc(IN_FILE);
			if(Abuf != NULL && Abuf[i] == 0x00){
				  Bbuf[i] = 0x00;
			}
			else Bbuf[i] = tmp;
		}
		  fclose(IN_FILE);

		  fprintf(OUT_FILE,"P6 352 288 255\r");
		for(int k=0;k<Height;k++){
			for(int j=0;j<Width;j++){
				fprintf(OUT_FILE,"%c%c%c",Rbuf[k*Width+j],Gbuf[k*Width+j],Bbuf[k*Width+j]);
			}
		}


		if(method){
			w = (unsigned int)Bbuf[Width*Height-2];
			if(w >= 11)
				w = 11;
		}
		int haha;
		ct1 == -1 ? ct1 = (unsigned int)Bbuf[Width*Height - 1] : ct2 == -1 ? ct2= (unsigned int)Bbuf[Width*Height - 1]:haha=1;

	  fclose(OUT_FILE);
	  Mat m=imread("source.ppm", mode);
	  return m;
}

void featureMatch(){

	Mat img_object = src_no_alpha;
	 Mat img_scene = candidate;

	  if( !img_object.data || !img_scene.data )
	  { std::cout<< " --(!) Error reading images " << std::endl; return; }

	  //-- Step 1: Detect the keypoints using SURF Detector
	  int minHessian = 3000;

	  SurfFeatureDetector detector( minHessian );

	  std::vector<KeyPoint> keypoints_object, keypoints_scene;

	  detector.detect( img_object, keypoints_object );
	  detector.detect( img_scene, keypoints_scene );

	  //-- Step 2: Calculate descriptors (feature vectors)
	  SurfDescriptorExtractor extractor;

	  Mat descriptors_object, descriptors_scene;

	  extractor.compute( img_object, keypoints_object, descriptors_object );
	  extractor.compute( img_scene, keypoints_scene, descriptors_scene );

	  if(ct1 < 4 && ct2 < 4 && ct1 != ct2){
		  cout<<ct1<<endl;
		  cout<<ct2<<endl;
		  cout<<"Feature match failed!no results !";
		  exit(0);
	  }
	  //-- Step 3: Matching descriptor vectors using FLANN matcher
	  FlannBasedMatcher matcher;
	  std::vector< DMatch > matches;
	  matcher.match( descriptors_object, descriptors_scene, matches );

	  double max_dist = 0; double min_dist = 100;

	  //-- Quick calculation of max and min distances between keypoints
	  for( int i = 0; i < descriptors_object.rows; i++ )
	  { double dist = matches[i].distance;
	    if( dist < min_dist ) min_dist = dist;
	    if( dist > max_dist ) max_dist = dist;
	  }

	  printf("-- Max dist : %f \n", max_dist );
	  printf("-- Min dist : %f \n", min_dist );

	  //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	  std::vector< DMatch > good_matches;

	  for( int i = 0; i < descriptors_object.rows; i++ )
	  { if( matches[i].distance < 3*min_dist )
	     { good_matches.push_back( matches[i]); }
	  }

	  Mat img_matches;
	  drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
	               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
	               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

	  //-- Localize the object
	  std::vector<Point2f> obj;
	  std::vector<Point2f> scene;

	  for( int i = 0; i < good_matches.size(); i++ )
	  {
	    //-- Get the keypoints from the good matches
	    obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
	    scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
	  }
	//  cout<<good_matches.size()<<endl;
//	  Mat H = findHomography( obj, scene, CV_RANSAC );

	  //-- Get the corners from the image_1 ( the object to be "detected" )
	  std::vector<Point2f> obj_corners(4);
	  obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
	  obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
	  std::vector<Point2f> scene_corners(4);

	//  perspectiveTransform( obj_corners, scene_corners, H);

	  //-- Draw lines between the corners (the mapped object in the scene - image_2 )
	  //line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
	  //line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
	  //line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
	 // line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );

	  //-- Show detected matches
	  //imshow( "Good Matches & Object detection", img_matches );
	  if(keypoints_object.size()<=5){
		  cout<<"not enough features detected in the source image! Ignore feature matching algorithm!"<<endl;
		  return;
	  }
}

int main(int argc, char * argv[])
{

	/// Read the image
	src_no_alpha = getImage(argv[2], NULL, 1, false);
	candidate = getImage( argv[1], NULL, 1, true);
	  if(argc == 4)
		  src = getImage( argv[2], argv[3], 1, false);
	  else
		  src = getImage( argv[2], NULL, 1, false);

	  Mat srct=src;

	  Hist_and_Backproj2();
	  search_x=p[w].xl;
	  search_y=p[w].yl;
	  bins=p[w].bv;
	  Mat cant=candidate;
	  if(w>2){
	  for(int i=0;i<cant.rows;i++){
		//  const unsigned char* Mi = src.ptr<unsigned char>(i);
		  for(int j=0;j<cant.cols;j++){
			  swap(cant.at<Vec3b>(i,j)[2],cant.at<Vec3b>(i,j)[1]);
		  }
	  }
	  for(int i=0;i<srct.rows;i++){
		//  const unsigned char* Mi = src.ptr<unsigned char>(i);
		  for(int j=0;j<srct.cols;j++){
			  swap(srct.at<Vec3b>(i,j)[2],srct.at<Vec3b>(i,j)[1]);
		  }
	  }
	  }

	  /// Transform it to HSV
	  cvtColor( src, hsv, CV_BGR2HSV );
	  cvtColor(candidate, hsv_candidate, CV_BGR2HSV);

	  /// Use only the Hue value
	  hue.create( hsv.size(), hsv.depth() );
	  int ch[] = { 0, 0 };
	  mixChannels( &hsv, 1, &hue, 1, ch, 1 );

	  hue_candidate.create(hsv_candidate.size(), hsv_candidate.depth());
	  mixChannels( &hsv_candidate, 1, &hue_candidate, 1, ch, 1 );

	  /// Create Trackbar to enter the number of bins
	  char* window_image = "Source image";
	  namedWindow( window_image, CV_WINDOW_AUTOSIZE );
	  char* window_candidate = "candidate";
	  namedWindow( window_candidate, CV_WINDOW_AUTOSIZE );
	  createTrackbar("* Hue  bins: ", window_image, &bins, 180, Hist_and_Backproj );
	  featureMatch();
	  Hist_and_Backproj(0, 0);

	  /// Show the image
	  if(w>2){
	  for(int i=0;i<cant.rows;i++){
		//  const unsigned char* Mi = src.ptr<unsigned char>(i);
		  for(int j=0;j<cant.cols;j++){
			  swap(cant.at<Vec3b>(i,j)[2],cant.at<Vec3b>(i,j)[1]);
		  }
	  }
	  for(int i=0;i<srct.rows;i++){
		//  const unsigned char* Mi = src.ptr<unsigned char>(i);
		  for(int j=0;j<srct.cols;j++){
			  swap(srct.at<Vec3b>(i,j)[2],srct.at<Vec3b>(i,j)[1]);
		  }
	  }
	  }
	  for(int i=biggest.y;i<biggest.y+search_y;i++){
		//  const unsigned char* Mi = src.ptr<unsigned char>(i);
		  for(int j=biggest.x;j<biggest.x+search_x;j++){
			  candidate.at<Vec3b>(i,j)[0]=255;
			  candidate.at<Vec3b>(i,j)[1]=0;
			  candidate.at<Vec3b>(i,j)[2]=0;

		  }
	  }

	  imshow( window_image, src );
	  imshow(window_candidate, candidate);

	  /// Wait until user exits the program
	  waitKey(0);
	  return 0;
}
/**
int main( int, char** argv )
{
  /// Read the image
  src = getImage( argv[2], argv[3] ,1 );
  candidate = getImage(argv[1], NULL, 1);
  /// Transform it to HSV
  cvtColor( src, hsv, COLOR_BGR2HSV );
  cvtColor(candidate, hsv_candidate, COLOR_BGR2HSV);

  namedWindow("candidate", WINDOW_AUTOSIZE);
  imshow( "candidate", candidate );
  /// Show the image
  namedWindow( "src", WINDOW_AUTOSIZE );
  imshow( "src", src );

  /// Set Trackbars for floodfill thresholds
  createTrackbar( "Low thresh", "src", &lo, 255, 0 );
  createTrackbar( "High thresh", "src", &up, 255, 0 );
  /// Set a Mouse Callback
  setMouseCallback( "src", pickPoint, 0 );
//pickPoint();
  waitKey(0);
  return 0;
}
**/
