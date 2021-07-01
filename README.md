# Generalization Comparison of Deep Neural Networks via Output Sensitivity

## Requirements
We conducted experiments under:   
- python 3.5.2. 
- torch 1.4.0. 
- torchvision 0.5.0. 
- cuda 10.2. 
- jupyter-notebook 6.0.3  
- ipython 7.9.0. 
- 1 Nvidia Titan X Maxwell GPU. 

Datasets:  
    MNIST, CIFAR-10 and CIFAR-100 will be automatically downloaded by running the script.  
    
## Description of files  
* datasets.py: includes the code to get data loader for MNIST, CIFAR-10 and CIFAR-100 datasets.  
* models.py: includes the code for neural network configurations that are used and two parameter initialization techniques.  
* experiments.py: includes the code to train the models and compute sensitivity in each epoch.  
* results.ipynb: includes the code to plot figures after the execution of experiments.py is finished.  
* temp.pkl: the saved results of the below example experiment. The setting of temp2.pkl and temp3.pkl differ from temp.pkl by changing the scale from 300 to 200 and 500, respectively.  

## Example 
To train a 4-layer FC with 300 hidden units, initialized using the standard normal distribution, on 1k points of the CIFAR-10 dataset, with batch size=128, run the following command:   
```
python3 experiments.py --model 'fc' --init SN --scale 300 --depth 4 --dataset cifar10 --numsamples 1000 --batchsize 128 --numepochs 2000 --filename <filename.pkl>
```
The results will be saved in <filename.pkl> file.  
To plot the figures of each experiment run results.ipynb file while reading the <filename.pkl> files of your choice.  

## Citation
To cite our work please use:
```
@inproceedings{forouzesh2021generalization,
  title={Generalization Comparison of Deep Neural Networks via Output Sensitivity},
  author={Forouzesh, Mahsa and Salehi, Farnood and Thiran, Patrick},
  booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},
  pages={7411--7418},
  year={2021},
  organization={IEEE}
}
```
or for the full arxiv version:
```
@misc{forouzesh2020generalization,
    title={Generalization Comparison of Deep Neural Networks via Output Sensitivity},
    author={Mahsa Forouzesh and Farnood Salehi and Patrick Thiran},
    year={2020},
    eprint={2007.15378},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```


