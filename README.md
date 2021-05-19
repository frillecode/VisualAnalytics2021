<br />

  <h1 align="center">Visual Analytics Exam 2021</h1> 

  <h2 align="center">Cultural Data Science, Aarhus University </h2> 
  <p align="center">
    Frida Hæstrup (201805753)
    <br>
</p>

<p align="center">
  <a href="https://github.com/frillecode/VisualAnalytics2021">
    <img src="figures/aulogo_uk_var2_blue.png" alt="Logo" width="200" height="50">
  </a>
<br>
<br>   

## Content
This is my personal repository with code and data related to my exam in the Spring 2021 module _Visual Analytics_ as part of the bachelor's tilvalg in [Cultural Data Science](https://bachelor.au.dk/en/supplementary-subject/culturaldatascience/) at Aarhus University. The portfolio contains 4 projects: 

| Project | Description|
|--------|:-----------|
1 | Finding text using Canny edge detection
2 | Classifier benchmarks using Logistic Regression and a Neural Network 
3 | Multi-class classification of impressionist painters 
4 | Classifying book genres using VGG-16


## Structure

This repository has the following directory structure:

```bash
VisualAnalytics2021/  
├── data/  #data folders for each project
│   └── project1/
│   └── project2/
│   └── project3/
│   └── project4/
├── src/  #Python scripts for each project
│   └── project1/
│   │   └── out/
│   │   └── edge_detection.py
│   └── project2/
│   │   └── out/
│   │   └── lr-mnist.py
│   │   └── nn-mnist.py 
│   └── project3/
│   │   └── out/
│   │   └── cnn-artists.py  
│   └── project4/
│   │   └── out/  
│   │   └── book_covers.py  
├── utils/  #utility functions 
│   └── *.py  
├── figures/  #figures to use in READMEs  
│   └── *.png  
```

## Technicalities
Scripts with code for each project can be found in the folder, _src_, along with a description of how to run them. 

To run scripts within this repository, I recommend cloning the repository and installing relevant dependencies in a virtual ennvironment:
```bash
$ git clone https://github.com/frillecode/VisualAnalytics2021
$ cd VisualAnalytics2021
$ bash ./create_venv.sh #use create_venv_win.sh for windows
```


If you run into issues with some libraries/modules not being installed correctly when creating the virtual environment, install these manually by running the following:
```bash
$ cd VisualAnalytics2021
$ source cds-vis/bin/activate
$ pip install {module_name}
$ deactivate
```

## Licencse
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
Credits for utility scripts and the original repository structure goes to [Ross Deans Kristensen-McLachlan](https://pure.au.dk/portal/en/persons/ross-deans-kristensenmclachlan(29ad140e-0785-4e07-bdc1-8af12f15856c).html).
