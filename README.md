# TransBTS_Python-implementation.-

Training and Testing (On Google Colab)

Video explanation link for training and testing setup: - https://drive.google.com/file/d/1XCvwjKsHp-PYli9JsqMQ-sO4sPvdcRK_/view?usp=sharing
Project GitHub repository link: - https://github.com/Wenxuan-1119/TransBTS

Step 1: - Download the Training dataset from this link: - https://drive.google.com/drive/folders/1wrgQote5E2FzyCJ8SaNH_d_DAh2jKULd?usp=sharing
Upload these files in a folder on your drive and name it ‘MICCAI_BraTS2020_TrainingData’

Step 2: - Download the Validation dataset from this link: - https://drive.google.com/drive/folders/1EZ8UXxgZT9eSHqMa6eINP1ghvdU19sLn?usp=sharing
Upload these files in a folder on your drive and name it ‘MICCAI_BraTS2020_ValidationData’

Step 3: - Download the training and testing files from this link: - https://drive.google.com/drive/folders/1cJ_8CzGzZKMfsyrSqiJ3PY0-ecEAkau8?usp=sharing
Upload these files in a folder on your drive and name it ‘TransBTS-main’

Step 4: - Open a new Google Colab notebook and link it to your drive.

Step 5: - Copy and paste the code from below which will download the necessary files and libraries for training and testing.

## Installation & Setup

1. **Library Installation**
    ```bash
    !pip uninstall torch 
    !pip install torch==1.6.0 torchvision 
    !pip install setproctitle 
    !pip install tensorboardx 
    !pip install pickle5 
    ```
    > **Note**: After installation, ensure you edit the `pickle` file as instructed below:
    Navigate to `/usr/local/lib/python3.7/dist-packages/pandas/io/pickle.py` and modify the first line to `import pickle5 as pickle`.

2. **Training**
    ```bash
    !python -m torch.distributed.launch --nproc_per_node=1 /content/drive/MyDrive/TransBTS-main/train.py --resume_dir '/content/drive/MyDrive/TransBTS-main/checkpoint/TransBTS2022-03-29/model_epoch_935.pth' --load=True --save_freq=1 --num_workers=2 --batch_size=4 --train_dir '/content/drive/MyDrive/MICCAI_BraTS2020_TrainingData' --valid_dir '/content/drive/MyDrive/MICCAI_BraTS2020_ValidationData' --train_file '/content/drive/MyDrive/TransBTS-main/train.txt' --valid_file '/content/drive/MyDrive/TransBTS-main/valid.txt'
    ```
    > **Note**: Remove `--resume_dir` and `--load` arguments if you are starting training from the first epoch!

3. **Post-Training**
    - After training completion, trained models will be saved in the `checkpoint` folder inside the `TransBTS` main directory.
    - Epoch logs can be found in the `log` directory of `TransBTS` main folder.

4. **Testing**
    - The validation dataset is utilized for testing (as there's no separate testing dataset for this competition).
    ```bash
    !python /content/drive/MyDrive/TransBTS-main/test.py --snapshot=False --post_process=False --load_file '/content/drive/MyDrive/TransBTS-main/checkpoint/TransBTS2022-03-29/model_epoch_935.pth' --num_workers=2 --output_dir='/content/drive/MyDrive/' --valid_dir '/content/drive/MyDrive/MICCAI_BraTS2020_ValidationData' --valid_file '/content/drive/MyDrive/TransBTS-main/valid.txt'
    ```
    > **Note**: Ensure the correct directory for the `--load_file` argument, pointing to the trained model in the `checkpoint` folder.

5. **Visualization and Submission**
    - Set the `--snapshot` argument to `True` to view model outputs in a `visualization` folder automatically created in your Google Drive.
    - Another folder named `submission` will be generated. After testing, upload files from `submission` to the UPENN website mentioned earlier to obtain dice scores.

## IMPORTANT NOTES

### Datasets
- The **TransBTS 2020** dataset original link: [https://ipp.cbica.upenn.edu/](https://ipp.cbica.upenn.edu/)
  - We obtained the dataset from the link after registering. Approval might take 2-3 days.
- Links for **KiTS19** and **LiTS17** datasets:
  - **KiTS19**: [https://github.com/neheller/kits19](https://github.com/neheller/kits19)
  - **LiTS17**: [https://competitions.codalab.org/competitions/17094](https://competitions.codalab.org/competitions/17094)
- Due to storage restrictions on Google Drive, we haven't uploaded the KiTS and LiTS datasets or their trained models like we did with Brats2020. You'll need to download, compress, and then upload these datasets yourself.
- The datasets (training + validation) are large, each totaling about 70GB or more. Due to space and bandwidth constraints, we employed a lossless compression algorithm instead of using pickle format. Further details are provided in the associated video.
- Training was predominantly executed on the Agave cluster platform. We set up our anaconda environment and then ran the training and testing files. Adjust dataset directory paths as needed. More details in the video.

### Model Code Files
- Relevant files: `IntmdSequential.py`, `PositionalEncoding.py`, `TransBTS_downsample8x_skipconnection.py`, `Transformer.py`, and `Unet_skipconnection.py`

### Data Compression & Conversion
1. Convert `.nii.gz` files to `.pkl`. This has been outlined on the GitHub repo. Run `preprocess.py` from the TransBTS main directory. Update the `root` directories accordingly in `preprocess.py`.
   - For **LiTS**: Unzip `.nii.zip` files first and then convert to `.nii.gz` using the gzip library before passing them to `preprocess.py`.
   - For **KiTS**: The dataset is in `.nii.gz` format by default.
2. Execute `data_compression.py` from the 'Codes' directory. This converts `.pkl` to `.bz2` (applies compression). Update directory paths as required.

> **Note**: Compression is primarily for data transfer across platforms (e.g., local machine, Google Colab, Agave Cluster). It doesn't influence training/testing performance or batch size. Annotated codes reside in the ‘Codes’ folder.

### Saving Intermediary Images
1. Substitute `TransBTS_downsample8x_skipconnection.py` in `TransBTS-main/models/` with our annotated version.
2. Run the training script as usual. Images will be saved to the specified directory.

> **Editing Tip**: For modifying saved images, navigate to line 229. This segment manages image creation. Adjust to modify image slices or to show encoded/decoded images.

---

*All the provided steps and code snippets, including library installation, training, and testing on Google Colab, are annotated and available in `train_test_codes_on_colab.py`.*
