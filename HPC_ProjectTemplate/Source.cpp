#include <iostream>
#include <math.h>
#include <stdlib.h>
#include<string.h>
#include<msclr\marshal_cppstd.h>
#include <omp.h>
#include <ctime>// include this header 
#include <mpi.h>
#pragma once

#using <mscorlib.dll>
#using <System.dll>
#using <System.Drawing.dll>
#using <System.Windows.Forms.dll>
using namespace std;
using namespace msclr::interop;

#define MODE 0
#define NUM_OF_THREADS 16

/*

*/

#define maxIntensity 255
#define newMaxIntensity 255

int* inputImage(int* w, int* h, System::String^ imagePath) //put the size of image in w & h
{
	int* input;


	int OriginalImageWidth, OriginalImageHeight;

	//*********************************************************Read Image and save it to local arrayss*************************	
	//Read Image and save it to local arrayss

	System::Drawing::Bitmap BM(imagePath);

	OriginalImageWidth = BM.Width;
	OriginalImageHeight = BM.Height;
	*w = BM.Width;
	*h = BM.Height;
	int *Red = new int[BM.Height * BM.Width];
	int *Green = new int[BM.Height * BM.Width];
	int *Blue = new int[BM.Height * BM.Width];
	input = new int[BM.Height*BM.Width];
	for (int i = 0; i < BM.Height; i++)
	{
		for (int j = 0; j < BM.Width; j++)
		{
			System::Drawing::Color c = BM.GetPixel(j, i);

			Red[i * BM.Width + j] = c.R;
			Blue[i * BM.Width + j] = c.B;
			Green[i * BM.Width + j] = c.G;

			input[i*BM.Width + j] = ((c.R + c.B + c.G) / 3); //gray scale value equals the average of RGB values

		}

	}
	return input;
}


void createImage(int* image, int width, int height, int index)
{
	System::Drawing::Bitmap MyNewImage(width, height);


	for (int i = 0; i < MyNewImage.Height; i++)
	{
		for (int j = 0; j < MyNewImage.Width; j++)
		{
			//i * OriginalImageWidth + j
			if (image[i*width + j] < 0)
			{
				image[i*width + j] = 0;
			}
			if (image[i*width + j] > 255)
			{
				image[i*width + j] = 255;
			}
			System::Drawing::Color c = System::Drawing::Color::FromArgb(image[i*MyNewImage.Width + j], image[i*MyNewImage.Width + j], image[i*MyNewImage.Width + j]);
			MyNewImage.SetPixel(j, i, c);
		}
	}
	MyNewImage.Save("..//Data//Output//outputRes" + index + ".jpg");
	cout << "result Image Saved " << index << endl;
}


int main()
{
	int ImageWidth = 4, ImageHeight = 4;

	int start_s, stop_s, TotalTime = 0;

	System::String^ imagePath;
	std::string img;
	img = "..//Data//Input//test2.jpg";

	imagePath = marshal_as<System::String^>(img);
	int* imageData = inputImage(&ImageWidth, &ImageHeight, imagePath);
	
	start_s = clock();

	//62
	#if MODE == 0

		int imageSize = ImageHeight * ImageWidth;


		int* pixelIntensitiesCount = new int[maxIntensity + 1];
		double* pixelIntensitiesProbability = new double[maxIntensity + 1];
		double* pixelIntensitiesCumulativeProbability = new double[maxIntensity + 1];
		double* newPixelIntensities = new double[newMaxIntensity + 1];

		// STEP 1
		for (int i = 0; i < maxIntensity + 1; i++) {
			pixelIntensitiesCount[i] = 0;
		}

		for (int i = 0; i < imageSize; i++) {
			pixelIntensitiesCount[imageData[i]]++;
		}

		// STEP 2
		for (int i = 0; i < maxIntensity + 1; i++) {
			pixelIntensitiesProbability[i] = (double)pixelIntensitiesCount[i] / (double)imageSize;
		}
	
		pixelIntensitiesCumulativeProbability[0] = pixelIntensitiesProbability[0];
		// STEP 3
		for (int i = 1; i < maxIntensity + 1; i++) {
			pixelIntensitiesCumulativeProbability[i] = pixelIntensitiesProbability[i] + pixelIntensitiesCumulativeProbability[i - 1];
		}

		// STEP 4
		double scalingFactor = newMaxIntensity / (maxIntensity);
		for (int i = 0; i < newMaxIntensity; i++) {
			newPixelIntensities[i] = pixelIntensitiesCumulativeProbability[i] * newMaxIntensity;
			newPixelIntensities[i] = floor(newPixelIntensities[i]);
		}

		// STEP 5
		int* elSooraElGedeeda = new int[imageSize];
		for (int i = 0; i < imageSize; i++) {
			elSooraElGedeeda[i] = newPixelIntensities[imageData[i]];
		}
	#endif

	// 27
	#if MODE == 1

		int imageSize = ImageHeight * ImageWidth;
		int* pixelIntensitiesCount = new int[maxIntensity + 1];
		double* pixelIntensitiesProbability = new double[maxIntensity + 1];
		double* pixelIntensitiesCumulativeProbability = new double[maxIntensity + 1];
		double* newPixelIntensities = new double[newMaxIntensity + 1];
		double scalingFactor = newMaxIntensity / (maxIntensity);
		int* elSooraElGedeeda = new int[imageSize];


		omp_set_num_threads(NUM_OF_THREADS);
	
		int tid;

		#pragma omp parallel private(tid)
		{
			tid = omp_get_thread_num();
			int* local = new int[maxIntensity + 1];

			// STEP 1
			#pragma omp for
			for (int i = 0; i < maxIntensity + 1; i++) {
				pixelIntensitiesCount[i] = 0;
			}
		
			for (int i = 0; i < maxIntensity + 1; i++) {
				local[i] = 0;
			}

			for (int i = (tid * imageSize) / NUM_OF_THREADS; i < ((tid + 1) * imageSize) / NUM_OF_THREADS; i++) {
				local[imageData[i]]++;
			}

			for (int i = 0; i < maxIntensity + 1; i++) {
				#pragma omp critical
				pixelIntensitiesCount[i] += local[i];
			}

			// STEP 2
			#pragma omp for
			for (int i = 0; i < maxIntensity + 1; i++) {
				pixelIntensitiesProbability[i] = (double)pixelIntensitiesCount[i] / (double)imageSize;
			}

			// STEP 3
			#pragma omp master
			{
				pixelIntensitiesCumulativeProbability[0] = pixelIntensitiesProbability[0];

				for (int i = 1; i < maxIntensity + 1; i++) {
					pixelIntensitiesCumulativeProbability[i] = pixelIntensitiesProbability[i] + pixelIntensitiesCumulativeProbability[i - 1];
				}
			}

			// STEP 4
			#pragma omp for
			for (int i = 0; i < newMaxIntensity; i++) {
				newPixelIntensities[i] = pixelIntensitiesCumulativeProbability[i] * newMaxIntensity;
				newPixelIntensities[i] = floor(newPixelIntensities[i]);
			}

			// STEP 5
			#pragma omp for
			for (int i = 0; i < imageSize; i++) {
				elSooraElGedeeda[i] = newPixelIntensities[imageData[i]];
			}

		}
	#endif

	#if MODE == 2

		int imageSize = ImageHeight * ImageWidth;
		int* pixelIntensitiesCount = new int[maxIntensity + 1];
		double* pixelIntensitiesProbability = new double[maxIntensity + 1];
		double* pixelIntensitiesCumulativeProbability = new double[maxIntensity + 1];
		double* newPixelIntensities = new double[newMaxIntensity + 1];

		MPI_Init(NULL, NULL);
		
		int* localData = new int[imageSize / NUM_OF_THREADS];

		MPI_Scatter(imageData, imageSize / NUM_OF_THREADS, MPI_INT, localData, imageSize / NUM_OF_THREADS, MPI_INT, 0, MPI_COMM_WORLD);


		// STEP 1
		for (int i = 0; i < maxIntensity + 1; i++) {
			pixelIntensitiesCount[i] = 0;
		}

		for (int i = 0; i < imageSize; i++) {
			pixelIntensitiesCount[imageData[i]]++;
		}

		// STEP 2
		for (int i = 0; i < maxIntensity + 1; i++) {
			pixelIntensitiesProbability[i] = (double)pixelIntensitiesCount[i] / (double)imageSize;
		}

		pixelIntensitiesCumulativeProbability[0] = pixelIntensitiesProbability[0];
		// STEP 3
		for (int i = 1; i < maxIntensity + 1; i++) {
			pixelIntensitiesCumulativeProbability[i] = pixelIntensitiesProbability[i] + pixelIntensitiesCumulativeProbability[i - 1];
		}

		// STEP 4
		double scalingFactor = newMaxIntensity / (maxIntensity);
		for (int i = 0; i < newMaxIntensity; i++) {
			newPixelIntensities[i] = pixelIntensitiesCumulativeProbability[i] * newMaxIntensity;
			newPixelIntensities[i] = floor(newPixelIntensities[i]);
		}

		// STEP 5
		int* elSooraElGedeeda = new int[imageSize];
		for (int i = 0; i < imageSize; i++) {
			elSooraElGedeeda[i] = newPixelIntensities[imageData[i]];
		}

		MPI_Finalize();

	#endif

	stop_s = clock();
	TotalTime += (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000;
	createImage(elSooraElGedeeda , ImageWidth, ImageHeight, 1);
	cout << "time: " << TotalTime << endl;
	
	free(imageData);
	return 0;

}



