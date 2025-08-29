import time
from flask import Flask, render_template, request, redirect
import os
import json
import random
import string
import boto3

app = Flask(__name__)
BUCKET = 'cloudcomputingpetsproject'
@app.route('/')
def index():
    pets = []
    pets = getPets(pets)
    return render_template('index.html', pets=pets)

@app.route('/upload', methods=['GET', 'POST'])
def upload ():
    if request.method == 'POST':
        pet_name = request.form['pet_name']
        pet_age = request.form['pet_age']
        pet_breed = request.form['pet_breed']
        image = request.files['file']
        #save the image and get the filename
        # image_filename = uploadImageOnFolder(image)
        image_filename = uploadImageOnS3(image)
        new_pet = {
            "name": pet_name,
            "age": pet_age,
            "breed": pet_breed,
            "image": image_filename
        }
        # Write to pets.json
        file1 = open("pets.json", "a")
        file1.write(json.dumps(new_pet) + "\n")
        file1.close()
        return redirect('/successfullUpload')
    return render_template('upload.html')


@app.route('/successfullUpload')
def successfullUpload ():
    return render_template('successfullUpload.html')


@app.route('/adoptPet')
def adoptPet():
    return render_template('adoptPet.html')


# Function to upload image on folder for local machine
def uploadImageOnFolder(image):
    timestamp = int(time.time())
    random_letters = ''.join(random.choices(string.ascii_letters, k=5))
    image_filename = f"{timestamp}_{random_letters}_{image.filename}"
    image.save(os.path.join('static', 'images', image_filename))
    return image_filename

def uploadImageOnS3(image):
#upload on s3
    s3 = boto3.client('s3')
    timestamp = int(time.time())
    random_letters = ''.join(random.choices(string.ascii_letters, k=5))
    image_filename = f"{timestamp}_{random_letters}_{image.filename}"
    s3.upload_fileobj(image,BUCKET,image_filename)
    url = f"https://{BUCKET}.s3.us-east-2.amazonaws.com/{image_filename}"
    return url

def getPets(pets)   :
    if os.path.exists('pets.json'):
        print("File exists")
        file1 = open("pets.json", "r")
        for line in file1:
            pets.append(json.loads(line))
        file1.close()
    else:
        print("File does not exist, nothing to read")
    return pets

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
    # app.run(debug=True)