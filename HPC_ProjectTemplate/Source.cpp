#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <msclr\marshal_cppstd.h>
#include <omp.h>
#include <ctime> // include this header
#include <mpi.h>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#pragma once

#using < mscorlib.dll>
#using < System.dll>
#using < System.Drawing.dll>
#using < System.Windows.Forms.dll>
using namespace std;
using namespace msclr::interop;

#define MODE 0
#define NUM_OF_THREADS 16
#define NUM_OF_PROCESSORS 4

#define maxIntensity 255
#define newMaxIntensity 255

int *inputImage(int *w, int *h, System::String ^ imagePath) // put the size of image in w & h
{
	int *input;

	int OriginalImageWidth, OriginalImageHeight;

	//*********************************************************Read Image and save it to local arrayss*************************
	// Read Image and save it to local arrayss

	System::Drawing::Bitmap BM(imagePath);

	OriginalImageWidth = BM.Width;
	OriginalImageHeight = BM.Height;
	*w = BM.Width;
	*h = BM.Height;
	int *Red = new int[BM.Height * BM.Width];
	int *Green = new int[BM.Height * BM.Width];
	int *Blue = new int[BM.Height * BM.Width];
	input = new int[BM.Height * BM.Width];
	for (int i = 0; i < BM.Height; i++)
	{
		for (int j = 0; j < BM.Width; j++)
		{
			System::Drawing::Color c = BM.GetPixel(j, i);

			Red[i * BM.Width + j] = c.R;
			Blue[i * BM.Width + j] = c.B;
			Green[i * BM.Width + j] = c.G;

			input[i * BM.Width + j] = ((c.R + c.B + c.G) / 3); // gray scale value equals the average of RGB values
		}
	}
	return input;
}

void createImage(int *image, int width, int height, int index)
{
	System::Drawing::Bitmap MyNewImage(width, height);

	for (int i = 0; i < MyNewImage.Height; i++)
	{
		for (int j = 0; j < MyNewImage.Width; j++)
		{
			// i * OriginalImageWidth + j
			if (image[i * width + j] < 0)
			{
				image[i * width + j] = 0;
			}
			if (image[i * width + j] > 255)
			{
				image[i * width + j] = 255;
			}
			System::Drawing::Color c = System::Drawing::Color::FromArgb(image[i * MyNewImage.Width + j], image[i * MyNewImage.Width + j], image[i * MyNewImage.Width + j]);
			MyNewImage.SetPixel(j, i, c);
		}
	}
	MyNewImage.Save("..//..//Data//Output//outputRes" + index + ".png");
	cout << "result Image Saved " << index << endl;
}

int main()
{
	int ImageWidth = 4, ImageHeight = 4;

	int start_s, stop_s, TotalTime = 0;

	System::String ^ imagePath;
	std::string img;
	img = "..//..//Data//Input//test.png";

	imagePath = marshal_as<System::String ^>(img);
	int *imageData = inputImage(&ImageWidth, &ImageHeight, imagePath);

#if MODE != 2
	start_s = clock();
#endif

#if MODE == 0

	int imageSize = ImageHeight * ImageWidth;

	int *pixelIntensitiesCount = new int[maxIntensity + 1];
	double *pixelIntensitiesProbability = new double[maxIntensity + 1];
	double *pixelIntensitiesCumulativeProbability = new double[maxIntensity + 1];
	double *newPixelIntensities = new double[newMaxIntensity + 1];

	// STEP 1
	for (int i = 0; i < maxIntensity + 1; i++)
	{
		pixelIntensitiesCount[i] = 0;
	}

	for (int i = 0; i < imageSize; i++)
	{
		pixelIntensitiesCount[imageData[i]]++;
	}

	// STEP 2
	for (int i = 0; i < maxIntensity + 1; i++)
	{
		pixelIntensitiesProbability[i] = (double)pixelIntensitiesCount[i] / (double)imageSize;
	}

	pixelIntensitiesCumulativeProbability[0] = pixelIntensitiesProbability[0];
	// STEP 3
	for (int i = 1; i < maxIntensity + 1; i++)
	{
		pixelIntensitiesCumulativeProbability[i] = pixelIntensitiesProbability[i] + pixelIntensitiesCumulativeProbability[i - 1];
	}

	// STEP 4
	double scalingFactor = newMaxIntensity / (maxIntensity);
	for (int i = 0; i < maxIntensity + 1; i++)
	{
		newPixelIntensities[i] = pixelIntensitiesCumulativeProbability[i] * newMaxIntensity;
		newPixelIntensities[i] = floor(newPixelIntensities[i]);
	}

	// STEP 5
	int *newImage = new int[imageSize];
	for (int i = 0; i < imageSize; i++)
	{
		newImage[i] = newPixelIntensities[imageData[i]];
	}
#endif

#if MODE == 1

	int imageSize = ImageHeight * ImageWidth;
	int *pixelIntensitiesCount = new int[maxIntensity + 1];
	double *pixelIntensitiesProbability = new double[maxIntensity + 1];
	double *pixelIntensitiesCumulativeProbability = new double[maxIntensity + 1];
	double *newPixelIntensities = new double[newMaxIntensity + 1];
	double scalingFactor = newMaxIntensity / (maxIntensity);
	int *newImage = new int[imageSize];

	omp_set_num_threads(NUM_OF_THREADS);

	int tid;

#pragma omp parallel private(tid)
	{
		tid = omp_get_thread_num();
		int *local = new int[maxIntensity + 1];

		// STEP 1
#pragma omp for
			for (int i = 0; i < maxIntensity + 1; i++)
			{
				pixelIntensitiesCount[i] = 0;
			}
		
		for (int i = 0; i < maxIntensity + 1; i++)
		{
			local[i] = 0;
		}

		for (int i = (tid * imageSize) / NUM_OF_THREADS; i < ((tid + 1) * imageSize) / NUM_OF_THREADS; i++)
		{
			local[imageData[i]]++;
		}

		#pragma omp barrier

		for (int i = 0; i < maxIntensity + 1; i++)
		{
#pragma omp critical
			pixelIntensitiesCount[i] += local[i];
		}
		

#pragma omp barrier
		// STEP 2
#pragma omp for
		for (int i = 0; i < maxIntensity + 1; i++)
		{
			pixelIntensitiesProbability[i] = (double)pixelIntensitiesCount[i] / (double)imageSize;
		}

		// STEP 3
#pragma omp master
		{
			pixelIntensitiesCumulativeProbability[0] = pixelIntensitiesProbability[0];

			for (int i = 1; i < maxIntensity + 1; i++)
			{
				pixelIntensitiesCumulativeProbability[i] = pixelIntensitiesProbability[i] + pixelIntensitiesCumulativeProbability[i - 1];
			}
		}

#pragma omp barrier

		// STEP 4
#pragma omp for
		for (int i = 0; i < maxIntensity + 1; i++)
		{
			newPixelIntensities[i] = pixelIntensitiesCumulativeProbability[i] * newMaxIntensity;
			newPixelIntensities[i] = floor(newPixelIntensities[i]);
		}

		// STEP 5
#pragma omp for
		for (int i = 0; i < imageSize; i++)
		{
			newImage[i] = newPixelIntensities[imageData[i]];
		}
	}
#endif

#if MODE == 2

	int imageSize = ImageHeight * ImageWidth;
	int *pixelIntensitiesCount = new int[maxIntensity + 1];
	double *pixelIntensitiesProbability = new double[maxIntensity + 1];
	double *pixelIntensitiesCumulativeProbability = new double[maxIntensity + 1];
	double *newPixelIntensities = new double[newMaxIntensity + 1];
	int *newImage = new int[imageSize];

	MPI_Init(NULL, NULL);
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	int *localImageData = new int[imageSize / NUM_OF_PROCESSORS];
	int localPixelIntensitiesCount[maxIntensity + 1];
	double localPixelIntensitiesProbability[maxIntensity + 1];
	double localPixelIntensitiesCumulataiveProbability[(maxIntensity + 1) / NUM_OF_PROCESSORS];
	double localNewPixelIntensities[(maxIntensity + 1) / NUM_OF_PROCESSORS];
	if (my_rank == 0)
	{
		start_s = clock();
	}

	// STEP 1
	MPI_Scatter(imageData, imageSize / NUM_OF_PROCESSORS, MPI_INT, localImageData, imageSize / NUM_OF_PROCESSORS, MPI_INT, 0, MPI_COMM_WORLD);

	for (int i = 0; i < maxIntensity + 1; i++)
	{
		localPixelIntensitiesCount[i] = 0;
	}

	for (int i = 0; i < imageSize / NUM_OF_PROCESSORS; i++)
	{
		localPixelIntensitiesCount[localImageData[i]]++;
	}

	MPI_Reduce(localPixelIntensitiesCount, pixelIntensitiesCount, maxIntensity + 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	// STEP 2
	MPI_Scatter(pixelIntensitiesCount, (maxIntensity + 1) / NUM_OF_PROCESSORS, MPI_INT, localPixelIntensitiesCount,
				(maxIntensity + 1) / NUM_OF_PROCESSORS, MPI_INT, 0, MPI_COMM_WORLD);

	for (int i = 0; i < (maxIntensity + 1) / NUM_OF_PROCESSORS; i++)
	{
		localPixelIntensitiesProbability[i] = (double)localPixelIntensitiesCount[i] / (double)imageSize;
	}

	double *newlocalPixelIntensitiesProbability = new double[maxIntensity + 1];
	MPI_Gather(localPixelIntensitiesProbability, (maxIntensity + 1) / NUM_OF_PROCESSORS, MPI_DOUBLE, newlocalPixelIntensitiesProbability, (maxIntensity + 1) / NUM_OF_PROCESSORS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	double *allPixelIntensitiesCumulataiveProbability = new double[maxIntensity + 1];
	if (my_rank == 0)
	{
		allPixelIntensitiesCumulataiveProbability[0] = newlocalPixelIntensitiesProbability[0];
		for (int i = 1; i < maxIntensity + 1; i++)
		{
			allPixelIntensitiesCumulataiveProbability[i] = newlocalPixelIntensitiesProbability[i] + allPixelIntensitiesCumulataiveProbability[i - 1];
		}
	}

	MPI_Scatter(allPixelIntensitiesCumulataiveProbability, (maxIntensity + 1) / NUM_OF_PROCESSORS, MPI_DOUBLE, localPixelIntensitiesCumulataiveProbability, (maxIntensity + 1) / NUM_OF_PROCESSORS, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	double scalingFactor = newMaxIntensity / (maxIntensity);

	for (int i = 0; i < (maxIntensity + 1) / NUM_OF_PROCESSORS; i++)
	{
		localNewPixelIntensities[i] = localPixelIntensitiesCumulataiveProbability[i] * newMaxIntensity;
		localNewPixelIntensities[i] = floor(localNewPixelIntensities[i]);
	}

	MPI_Allgather(localNewPixelIntensities, (maxIntensity + 1) / NUM_OF_PROCESSORS, MPI_DOUBLE, newPixelIntensities, (maxIntensity + 1) / NUM_OF_PROCESSORS, MPI_DOUBLE, MPI_COMM_WORLD);

	int *localnewImage = new int[imageSize / NUM_OF_PROCESSORS];
	for (int i = 0; i < imageSize / NUM_OF_PROCESSORS; i++)
	{
		localnewImage[i] = newPixelIntensities[localImageData[i]];
	}

	MPI_Gather(localnewImage, imageSize / NUM_OF_PROCESSORS, MPI_INT, newImage, imageSize / NUM_OF_PROCESSORS, MPI_INT, 0, MPI_COMM_WORLD);

	if (my_rank == 0)
	{
		stop_s = clock();
		TotalTime += (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000;
		createImage(newImage, ImageWidth, ImageHeight, 1);
		cout << "time: " << TotalTime << endl;

		cv::Mat image = cv::imread("..//..//Data//OutPut//outputRes1.png", cv::IMREAD_GRAYSCALE);

		int histSize = 256;
		float range[] = { 0, 256 };
		const float* histRange = { range };
		cv::Mat hist;

		cv::calcHist(&image, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
		cv::normalize(hist, hist, 0, 400, cv::NORM_MINMAX);

		int histWidth = 512, histHeight = 400;
		int binWidth = cvRound((double)histWidth / histSize);
		cv::Mat histImage(histHeight + 50, histWidth + 50, CV_8UC3, cv::Scalar(255, 255, 255));  // White background

		cv::line(histImage, cv::Point(40, 0), cv::Point(40, histHeight), cv::Scalar(0, 0, 0), 2); // Y-axis
		cv::line(histImage, cv::Point(40, histHeight), cv::Point(histWidth + 40, histHeight), cv::Scalar(0, 0, 0), 2); // X-axis

		for (int i = 0; i < histSize; i++) {
			int height = cvRound(hist.at<float>(i));
			cv::Scalar color = cv::Scalar(i % 256, 255 - i % 256, (i * 2) % 256);  // Generate unique color

			cv::rectangle(histImage,
				cv::Point(40 + i * binWidth, histHeight - height),
				cv::Point(40 + (i + 1) * binWidth, histHeight),
				cv::Scalar(0, 0, 0),
				cv::FILLED);
		}

		for (int i = 0; i <= 255; i += 32) {
			std::string label = std::to_string(i);
			int xPos = 40 + i * binWidth;
			cv::putText(histImage, label, cv::Point(xPos, histHeight + 20), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1, 8);
		}


		for (int i = 0; i <= 400; i += 50) {
			std::string label = std::to_string(i);
			cv::putText(histImage, label, cv::Point(10, histHeight - i), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1, 8);
		}


		cv::imshow("Histogram", histImage);
		cv::waitKey(0);



		free(imageData);
	}

	MPI_Finalize();

#endif

#if MODE != 2
	stop_s = clock();
	TotalTime += (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000;
	createImage(newImage, ImageWidth, ImageHeight, 1);
	cout << "time: " << TotalTime << endl;


	cv::Mat image = cv::imread("..//..//Data//OutPut//outputRes1.png", cv::IMREAD_GRAYSCALE);

	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	cv::Mat hist;

	cv::calcHist(&image, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
	cv::normalize(hist, hist, 0, 400, cv::NORM_MINMAX);

	int histWidth = 512, histHeight = 400;
	int binWidth = cvRound((double)histWidth / histSize);
	cv::Mat histImage(histHeight + 50, histWidth + 50, CV_8UC3, cv::Scalar(255, 255, 255));  // White background

	cv::line(histImage, cv::Point(40, 0), cv::Point(40, histHeight), cv::Scalar(0, 0, 0), 2); // Y-axis
	cv::line(histImage, cv::Point(40, histHeight), cv::Point(histWidth + 40, histHeight), cv::Scalar(0, 0, 0), 2); // X-axis

	for (int i = 0; i < histSize; i++) {
		int height = cvRound(hist.at<float>(i));
		cv::Scalar color = cv::Scalar(i % 256, 255 - i % 256, (i * 2) % 256);  // Generate unique color

		cv::rectangle(histImage,
			cv::Point(40 + i * binWidth, histHeight - height),
			cv::Point(40 + (i + 1) * binWidth, histHeight),
			cv::Scalar(0,0,0),
			cv::FILLED);
	}

	for (int i = 0; i <= 255; i += 32) {  
		std::string label = std::to_string(i);  
		int xPos = 40 + i * binWidth;
		cv::putText(histImage, label, cv::Point(xPos, histHeight + 20), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1, 8);
	}

		
	for (int i = 0; i <= 400; i += 50) {  
		std::string label = std::to_string(i);  
		cv::putText(histImage, label, cv::Point(10, histHeight - i), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1, 8);
	}

		
	cv::imshow("Histogram", histImage);
	cv::waitKey(0);




	free(imageData);

#endif

	return 0;
}
