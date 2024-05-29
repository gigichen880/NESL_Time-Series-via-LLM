import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
from chronos import ChronosPipeline


def npy2csv(dir, new_dir):
  # if new_dir not exist, use the next line
  # os.makedirs(new_dir)

  # Loop through all files in the directory
  for filename in os.listdir(dir):
  # Load the npy file
    data = np.load(dir+str(filename))
    # Save the data to a CSV file
    new_filename = filename.replace(".npy", ".csv")
    path = new_dir+new_filename
    np.savetxt(path, data, delimiter=',')

def handleCol(csvPath, indexStr, isGt):
  if isGt:
     path = csvPath + indexStr + "_gt.csv"
  else:
    path = csvPath + indexStr + ".csv"
  dt = pd.read_csv(path)
  curr_col = dt.columns[0]
  dt.loc[len(dt)] = {curr_col: 0}
  dt = dt.shift(1)
  dt.iloc[0, 0] = float(curr_col)
  dt = dt.rename(columns={curr_col: indexStr})
  return dt

def toSingleCsv(new_dir, isGt, lengths, exp_size):

  for length in lengths: 
      for i in range(exp_size):
          new_col_name = str(i) + "_" + str(length)
          if i == 0:
              result = handleCol(new_dir, new_col_name, isGt)
              continue
          final_t = handleCol(new_dir, new_col_name, isGt)
          result = pd.concat([result, final_t], axis=1)
          if isGt:
            new_path = "ecg_" + str(length) + "_gt_dataset.csv"
          else:
             new_path = "ecg_" + str(length) + "_dataset.csv"
          result.to_csv(new_path, index=False)



def visualize(id, exp_size, test_length):

  for length in test_length:
      result = pd.read_csv("ecg_" + str(length) + "_dataset.csv")
      li = []
      gt_data = pd.read_csv("ecg_" + str(length) + "_gt_dataset.csv")
      for col in range(exp_size):
          sample = result.iloc[:, col].dropna()
          li.append(torch.tensor(sample))

      pipeline = ChronosPipeline.from_pretrained(
          "amazon/chronos-t5-small",
          device_map="cpu",  # use "cpu" for CPU inference and "mps" for Apple Silicon
          torch_dtype=torch.bfloat16,
      )
      # context must be either a 1D tensor, a list of 1D tensors,
      # or a left-padded 2D tensor with batch as the first dimension
      context = li
      prediction_length = length
      forecast = pipeline.predict(
          context,
          prediction_length,
          num_samples=20,
          temperature=1.0,
          top_k=50,
          top_p=1.0,
      ) # forecast shape: [num_series, num_samples, prediction_length]
      

      for i in range(exp_size):
          # visualize the forecast
          col_name = str(i) + "_" + str(length)
          forecast_index = range(len(context[i]), len(context[i]) + prediction_length)
          low, median, high = np.quantile(forecast[i].numpy(), [0.1, 0.5, 0.9], axis=0)

          plt.figure(figsize=(8, 4))
          plt.plot(result.iloc[:, i].dropna(), color="royalblue", label="historical data")
          gt = gt_data.iloc[:, i]
          gt.index+=len(context[i])
          plt.plot(gt, color="green", label="ground truth")
          
          
          plt.plot(forecast_index, median, color="tomato", label="median forecast")
          plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
          plt.legend()
          plt.title("Chronos-forecasting on ECG Data: " + col_name)
          plt.grid()

          plt.show()
          new_dir = "./" + id + "_visuals/"
          os.makedirs(new_dir)
          path = new_dir + col_name + '.jpg'
          plt.savefig(path)



def main():
  
  target = "ecg" # or ppg
  dir = "./" + target + "-extrapolation/" # please make sure the relative path between data and this file is correct
  new_dir = './' + target + '_ex_csv/' # where to store combined csv file 
  lengths = [5, 10]  # [5, 10, 25, 50, 100]
  exp_size = 20 # number of experiments wanted
  npy2csv(dir, new_dir) # convert original separate npy file to separate csv file
  toSingleCsv(new_dir, False, lengths, exp_size) # convert into one test dataset per test length & target (eg. ecg_5 or ppg_50)
  toSingleCsv(new_dir, True, lengths, exp_size) # convert into one ground truth dataset per test length & target (eg. ecg_5_gt or ppg_50_gt)
 # if execute once does not work, first comment the visualize and execute above, then finally call visualize()
  visualize(target, exp_size, lengths)

if __name__ == "__main__":

  main()