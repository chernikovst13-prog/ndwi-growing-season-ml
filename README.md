Overview
This project uses Machine Learning to determine growing (vegetation) seasons by predicting smoothed temperatures based on the Normalized Difference Water Index (NDWI) over the course of a year. This approach is especially useful when direct temperature data is unavailable due to a lack of local weather stations.

Instructions
1. Setting Up the Training Data
To train the model, we use a ready-made dataset of temperature and NDWI index values obtained from a weather station and satellite imagery in the village of Petrun (Komi Republic, Russia).

The .txt files contain the temperature data.  

The .csv files contain the NDWI index data.  

All of these files make up the training sample.  

After downloading and extracting the dataset, you need to specify the path to this folder in the code. Because all the training files are located in the same folder, you can use the exact same path for both the temperature and NDWI training directories:

Important Note: Each file in the training directory must represent exactly 1 year of data.

2. Setting Up the Test Data
In the test_ndwi_dir variable, specify the path to the folder containing the .csv files with the NDWI indices for which you want to determine the onset dates of the vegetation seasons (growing seasons).

The test file itself must be formatted strictly to contain only the date (Year-Month-Day) and the NDWI index, as shown below:

![CSV Format]

3. Setting Up the Output Directory
In the output_directory variable, simply indicate the path where the resulting vegetation season dates and generated graphs will be saved.

4. Running the Program
After launching the program, a prompt will appear in the terminal asking you to define the moving average (smoothing) window for the temperature values in the training sample. It is highly recommended to input the number 3.

5. Reviewing the Results
For each year processed, the program will output a graph showing the predicted temperature trend and the start/end dates of the vegetation seasons.

Along with the graphs, the program will generate 4 files in your output directory:

all_ndvi (or petrun_all_ndvi): Represents the temperature trend recorded at the training station.

ml_predicted_details: Contains the predicted daily temperature for each day, calculated based on the obtained NDWI index.

my_training_data: The dataset compiled for the model so it can understand the dependencies between the NDWI and the temperature.

predicted_vegetation_seasons: Contains the final predicted dates for the onset of the vegetation seasons.
