# Machine Learning API
After you trained a machine learning model,
you might want someone to test or deploy it on your product. 

This repository contains the sample code to make your machine learning API from the scratch. 

Here, FastAPI and Gender Detection Model are used. 

## Usage 
### Test on your local 
1. Create your original env by using `pyenv` or `venv`
- pyenv
  ```
  >> /Users/USERNAME/.pyenv/versions/3.6.5/bin/python -m venv venv/3.8.10
  >> source venv/3.6.5/bin/activate
  >> (3.6.5) pip install --upgrade pip
  ```
- venv
  ```
  >> python -m venv venv/3.6.5
  >> source venv/3.6.5/bin/activate
  >> (3.6.5) pip install --upgrade pip
  ```
2. Install packages
```
>> pip install -r requirements.txt
```

3. Run!
```
>> cd src
>> python app.py
```

4. POST an image to the endpoint `http://0.0.0.0:7000/genpredict/`

### Convert to Docker Image
1. Build a image 
```
>> pwd
~/ml-api/src
>> cd ..
>> sh build.sh
```
Once it's done, `mlapi` image was created. 

2. Create the container
```

```

