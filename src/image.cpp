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
using namespace cv;
using namespace std;


/// Global Variables
Mat src; Mat hsv; Mat hue;
Mat candidate; Mat hsv_candidate; Mat hue_candidate;
int bins = 36;
int search_x=110,search_y=90;

struct locf{
	int sum,x,y;
	locf(int s,int a,int b){ sum=s;x=a,y=b;}
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
	    	  sum+=cur;
	    	  //cout<<cur<<" ";
	      }
	     // cout<<endl;
	  }

	return sum;



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

Mat getImage(char* filepath, char* alphafile, int mode){
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

	  fclose(OUT_FILE);
	  Mat m=imread("source.ppm", mode);
	  return m;
}



int main(int argc, char * argv[])
{

	/// Read the image
	  if(argc == 4)
		  src = getImage( argv[2], argv[3], 1 );
	  else
		  src = getImage( argv[2], NULL, 1 );
	  candidate = getImage( argv[1], NULL, 1);
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
	  Hist_and_Backproj(0, 0);

	  /// Show the image

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

