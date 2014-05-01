/*
 * image.cpp
 *
 *  Created on: Apr 29, 2014
 *      Author: zhengwang
 */


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
using namespace cv;


Mat getImage(char* filepath){
	  FILE* IN_FILE; // fb.raw
	  FILE* OUT_FILE; // fb.ppm
	  unsigned char red, green, blue; // 8-bits each
	  unsigned int maxval; // max color val
	  unsigned short Width, Height;
	  size_t i;

	  IN_FILE = fopen(filepath, "rb");
	  OUT_FILE = fopen("source.ppm", "wb");
	  Width = 352;
	  Height = 288;
	  maxval = 255;

	  // P3 - PPM "plain" header
	  //fprintf(outfile, "P3\n#created with rgb2ppm\n%d %d\n%d\n", width, height, maxval);


	  char *Rbuf = new char[Height*Width];
	  char *Gbuf = new char[Height*Width];
	  char *Bbuf = new char[Height*Width];

		for (i = 0; i < Width*Height; i ++)
		{
			Rbuf[i] = fgetc(IN_FILE);
		}
		for (i = 0; i < Width*Height; i ++)
		{
			Gbuf[i] = fgetc(IN_FILE);
		}
		for (i = 0; i < Width*Height; i ++)
		{
			Bbuf[i] = fgetc(IN_FILE);
		}
		  fclose(IN_FILE);

		  fprintf(OUT_FILE,"P6 352 288 255\r");
		for(int k=0;k<Height;k++){
			for(int j=0;j<Width;j++){
				fprintf(OUT_FILE,"%c%c%c",Rbuf[k*Width+j],Gbuf[k*Width+j],Bbuf[k*Width+j]);
			}
		}

	  fclose(OUT_FILE);
	  Mat m=imread("source.ppm",1);
	  return m;
}


int main(int argc, char * argv[])
{

  Mat m= getImage(argv[1]);
  imshow("noway ",m);
   waitKey(0);
    return 0;
}

