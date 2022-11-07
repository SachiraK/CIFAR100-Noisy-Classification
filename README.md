# CIFAR100-Noisy-Classification

To train the noisy CIFAR-100 dataset, please run the following scripts in order.

1. python create_data.py --csv_file data/cifar100_nl.csv --thresh 0.5
   This will create a new dataset with removing the outliers of each class of the noisy labeled dataset.

2. python main.py --model DLA --num_epochs 100 --csv_file NewData.csv
   This will train the model on the specified feature extractor. Possible choices for the model are 'ResNet18', 'ResNet50', 'ResNet101' and 'DLA'.

To obtain the test accuracy of the model, please run the following script.
python test.py --csv_file data/cifar100_nl_test.csv --load_path {PATH OF THE SAVED TRAINED MODEL} --model {MODEL USED FOR TRAINING}
