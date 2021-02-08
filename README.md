# AlmondsClassifier
AlmondsClassifier using tensorflow and open cv 

### Frameworks 
 * Keras 2.0.8
 * Tensorflow 1.15.0
 * TfLearn 0.3.2
 * OpenCv-Python 4.4.0.42
 
# Main 
 You can run the main function to start training of mode of your choice by passing the name as args 
 
 # Export Model for serving 
 Run ` python3 export_model.py  <mode h5 path> <location to the export path>/1`
 
 #Serve the model 
 * Run the base image 
 ` docker run -d --name serving_base tensorflow/serving `
 * export the path of the exported model 
    ` export MODEL_PATH=<> `
 * Copy the model to you container (you can copy as many models as you want , just specify the endpoint correctly)
 `docker cp ${MODEL_PATH} serving_base:/models/alexnet`   
 
 * Commit the changes so docker will generate the new docker image  (change the names according to you)
  `docker commit --change "ENV MODEL_NAME alexnet" serving_base alexnet `
 
 * Kill the base container 
  `docker kill serving_base `
 
 * Run your new Image 
  `docker run -p 8501:8501  alexnet` 
  
 * The image will expose endpoint as following : 
 `http://localhost:8501/v1/models/alexnet:predict` 
  
#Versions of the model 
* v1 5 classes 200 iteration 94 percent 
* v2_1 16 class 110 iteration 74 percent 
* v2_2 16 class 200 iteration
