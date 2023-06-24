# Model
  The model is little different from the previous one. Since, there are multiple frames that are being tracked, this makes the `Input Data` of variable size. `Padding` is applied with pre processing as well as `Masking` while training the model. `LSTM` is used to eliminate masking and process with the remaining data.

  Since, the dataset is small, `batch_size` is kept `16` for optimal performance. The rest of the model is same as [before](https://github.com/SAM-DEV007/Instagram-Filters/blob/main/Hand_Gesture/Model), but is not converted to `TfLite`.

# Dataset
  The dataset uses the same principle as with the one before. All the data is passed in `1D array` for the captured multiple frames. The `0` obtained in pre-processing is converted to `0.0001` to prevent getting masked. Although, one column is unutilized (column B) which have the data on no. of columns of captured data.
